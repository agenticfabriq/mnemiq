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
    `select_definitions` matches inflection-tolerantly -- `term_pattern` appends `\\w*` to the
    last word, which is how "premium" retrieves on a question saying "premiums" -- while this check
    compared for exact equality. So retrieval puts `total payment` in the packet BECAUSE the model
    said "total payments", and the guard then reports "total payments" ungrounded and refuses.

    A false deferral on a defined term, sourced in a lexical detail rather than in anything about
    meaning. It matters beyond the one answer: the withdrawal rule for this guard is a measured
    over-declaration rate, so a matcher bug spending that budget retires a guard for a reason that
    was never about the design.
    """
    assert ungrounded_terms([spelling], _defs()) == []


@pytest.mark.parametrize("declared,certified", [("policies", "policy"), ("entities", "entity")])
def test_the_plural_no_suffix_rule_reaches(declared, certified):
    """`policy` -> `policies` loses a character off the stem, so no suffix appended to the literal
    can produce it. Neither width covered it and each was wrong in its own direction: retrieval
    missed the definition on a question saying "policies", and grounding refused "policies" with
    `policy` certified -- a deferral of an answerable question, which is the cost this guard is
    measured on. Business vocabularies are made of these words."""
    from mnemiq.contract.semantic import Definition

    defs = [Definition(id=f"fspay:policy:{certified}", term=certified, domain="fspay",
                       definition=f"the certified {certified}")]
    assert ungrounded_terms([declared], defs) == []


def test_a_longer_phrase_is_not_grounded_by_the_term_it_contains():
    """The limit on that tolerance, and the reason it is a full match rather than a search.

    "revenue per customer" is a DERIVATION over a defined term, not the term -- exactly the M35
    shape, where every part is certified and the composition is not. Inflection tolerance that
    matched a substring would ground it and silence the guard on its own finding.
    """
    assert ungrounded_terms(["revenue per customer"], _defs()) == ["revenue per customer"]
    assert ungrounded_terms(["lifetime value of revenue"], _defs()) == ["lifetime value of revenue"]


@pytest.mark.parametrize(
    "declared,certified",
    [("policyholder", "policy"), ("claimant", "claim"), ("revenuer", "revenue")],
)
def test_a_word_that_merely_starts_with_a_defined_term_is_not_that_term(declared, certified):
    """The other half of the tolerance, and the one that fails silently.

    Sharing retrieval's matcher meant sharing its `\\w*`, which extends the last word without
    limit. In retrieval that widens RECALL and the cost is a spare definition the model ignores.
    Here it widens GROUNDING: `policy` grounds `policyholder`, so a model that declared
    "policyholder" against a corpus certifying only "policy" is handed the confident answer this
    guard exists to refuse -- and it is handed it with no deferral, no reason and nothing in the
    result to read.

    The phrase limit does not cover this. "revenue per customer" is blocked because the extension
    cannot cross a space; `policyholder` needs no space. So grounding matches on a plural and
    nothing else, while retrieval keeps the wide rule it can afford.
    """
    from mnemiq.contract.semantic import Definition

    defs = [Definition(id=f"fspay:policy:{certified}", term=certified, domain="fspay",
                       definition=f"the certified {certified}")]
    assert ungrounded_terms([declared], defs) == [declared]
    assert ungrounded_terms([certified], defs) == [], "the term itself still grounds"


def test_every_ungrounded_term_survives_deduplication():
    """The refusal is a repair instruction, so it names all of them. Routing dedup through the
    same comparison is what makes that fragile: under the wide rule `policyholder` deduplicated
    against `policy` and the caller was told to define one of the two terms they must define."""
    assert ungrounded_terms(["policy", "policyholder"], []) == ["policy", "policyholder"]
    assert ungrounded_terms(["revenue", "revenues"], []) == ["revenue"], (
        "one term spelled two ways is still one term"
    )


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
    bound = [d.model_copy(update={"bound_objects": ["fs.payments"]}) for d in _defs()] + [
        # `policy` -> `policies` is the pairing that had NEITHER matcher: retrieval missed the
        # definition and grounding refused the term. Listed here rather than only in its own test
        # because this is the test whose job is that the two stay agreed -- dropping the `-ies`
        # branch from retrieval's width alone must not leave the suite green.
        Definition(id="fspay:policy:policy", term="policy", domain="fspay",
                   definition="a written contract", bound_objects=["fs.payments"]),
    ]
    grants = GrantSet(objects=frozenset({"fs.payments"}))
    for spelling in ("total payments", "revenues", "payment dates", "policies"):
        retrieved = select_definitions(spelling, bound, grants)
        assert retrieved, f"retrieval finds a definition for {spelling!r}"
        assert ungrounded_terms([spelling], retrieved) == [], (
            f"retrieval matched {spelling!r} but the guard called it ungrounded"
        )


def test_an_id_is_a_name_only_for_a_definition_that_has_no_other():
    """Which name a definition answers to, and why a term shuts the id out.

    A record may carry its name in `id` alone, so the tail has to be a spelling for those -- but
    only for those. The tail was briefly a spelling ALONGSIDE the term, and that is the M35
    suppressing direction: `term="net revenue"` with `id="...:revenue"` certifies net revenue, so
    grounding a model's declared "revenue" on it hands back the confident answer for a term the
    definition does not define. Where a term exists it is the certified name; the id is an
    implementation detail that happens to be legible.

    The whole namespaced id is never a name, however spelled -- nobody writes it into a sentence.
    """
    from mnemiq.contract.semantic import Definition

    named = [Definition(id="fspay:policy:x9", term="settled volume", domain="fspay",
                        definition="volume of settled payments")]
    assert ungrounded_terms(["settled volume"], named) == [], "the term is the name"
    assert ungrounded_terms(["x9"], named) == ["x9"], (
        "a definition that HAS a term is not also named by its id"
    )

    unnamed = [Definition(id="fspay:policy:loss_ratio", term="", domain="fspay",
                          definition="incurred losses over earned premium")]
    assert ungrounded_terms(["loss ratio"], unnamed) == [], "with no term, the tail is the name"
    assert ungrounded_terms(["fspay:policy:loss_ratio"], unnamed) == ["fspay:policy:loss_ratio"], (
        "a namespaced id is not something a model writes"
    )


def test_a_disagreeing_id_cannot_ground_a_term_its_definition_does_not_certify():
    """The failure the narrowing prevents, stated as the answer it would have returned.

    Certified: `net revenue`. Asked about: `revenue`. Those are different quantities and the
    definition says so; a guard that treats the id tail as a second certified name would let the
    model's invented plain-revenue derivation through with a certified-looking definition behind
    it, which is worse than no guard -- it is the guard vouching for the guess.
    """
    from mnemiq.contract.semantic import Definition

    defs = [Definition(id="fspay:policy:revenue", term="net revenue", domain="fspay",
                       definition="revenue net of refunds and chargebacks")]
    assert ungrounded_terms(["revenue"], defs) == ["revenue"]
    assert ungrounded_terms(["net revenue"], defs) == []


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

    out = plan_query(packet, snapshot, GrantSet(frozenset({"payment"})), _Gen(),
                     guard_undefined_terms=True)
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
        guard_undefined_terms=True,
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

    body = _guided_extra_body(True, declare_assumed_terms=True)
    schema = body["response_format"]["json_schema"]["schema"]
    assert "assumed_terms" in schema["properties"], "the sent schema must carry it, not just ours"
    assert "assumed_terms" not in schema.get("required", []), (
        "declaring nothing is the common case and must stay valid"
    )


def test_the_prompt_asks_for_the_field_the_parser_reads():
    """The prompt names the key, the schema permits it and the parser reads it -- three places,
    nothing tying them together. Rename one and the guard goes silently inert: the model sends a
    field nobody reads, or the parser waits for a field nobody asked for. The same
    hand-enumerated-allowlist shape that caused the finding this guard exists for.

    Each leg asserted against the thing that carries the key, never a substring of a module. The
    first version was `"assumed_terms" in inspect.getsource(generator)`, and the same diff put that
    literal in three places -- schema, dataclass field, parser -- so renaming the parser's
    `payload.get` alone left it green. `prompts` has the same problem for the same reason: the key
    appears in the JSON reply TEMPLATE and again in the prose explaining it, and only the template
    rename stops the model emitting the field. So the prompt leg reads the rendered template line,
    not the module text.
    """
    import json

    from mnemiq.generate import prompts
    from mnemiq.generate.generator import _GUIDED_SQL_SCHEMA

    key = "assumed_terms"
    rendered = prompts.system_prompt(dialect="duckdb", declare_assumed_terms=True)
    # The template that carries SQL, not the `{defer}` block's own `{"sql": null, ...}` -- there
    # are two, and the first one in the prompt is the deferral shape, which needs no declaration.
    template = next((ln for ln in rendered.splitlines() if "<the SELECT" in ln), None)
    assert template, "the prompt must show a JSON reply template"
    assert key in template, "the reply template must ask for it"
    assert key in _GUIDED_SQL_SCHEMA["properties"], "a guided reply must be permitted to carry it"
    carried = _parse(json.dumps({"sql": "SELECT 1", key: ["lifetime value"]}))
    assert carried.assumed_terms == ["lifetime value"], (
        "the parser must read the key the prompt asks for"
    )


# -- withdrawn on its own criterion ----------------------------------------------------------------


def test_the_guard_is_off_unless_asked_for():
    """The measurement, as a test, so the default cannot drift back silently.

    Beacon's answerable band at `30632b9`, one pass over 24 items (run
    `06a95802-d794-7a2b-8000-ead4e3573b73`): **12 deferrals**, against a prior of 0 in 144
    observations. Strict accuracy 66.7% -> 45.8%. The threshold was 2-3%, written down before the
    number was seen, together with "pull the guard rather than tune it".

    The model declared `data` ("which currencies appear in the data"), `processed`, `take in`,
    `fourth quarter of 2025` and `merchants` as business terms requiring a certified definition.
    The 2026-08-13 review killed the DETERMINISTIC version of this rule on the sentence "two
    consecutive words absent from a schema vocabulary is the normal condition of a sentence, not a
    signal", and the same failure arrived through the model-declared version built to route around
    it. That is the design's premise -- that declaring is a question the model can answer reliably
    where refusing is not -- and the band falsified it.

    So the same reply that defers under the flag answers without it. Turning the default back on
    means a new measurement, not a prompt tweak.
    """
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Column, Snapshot
    from mnemiq.generate.generator import FakeGenerator
    from mnemiq.generate.plan_query import Deferred, plan_query
    from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

    reply = '{"sql": "SELECT n FROM payment", "assumed_terms": ["lifetime value"]}'
    packet = ContextPacket(
        question="What is the lifetime value of our average customer?",
        cards=[RetrievedCard(object_id="payment", card="TABLE payment", score=1.0)],
        grant_fingerprint="f", enrichment_version="v1", definitions=_defs(),
    )
    snapshot = Snapshot(version="v1", source_id="fs", created_at="t",
                        columns=[Column(id="payment.n", object_id="payment", name="n")])
    grants = GrantSet(frozenset({"payment"}))

    on = plan_query(packet, snapshot, grants, FakeGenerator([reply]), guard_undefined_terms=True)
    assert isinstance(on, Deferred) and on.code == DeferralReason.UNDEFINED_TERM

    off = plan_query(packet, snapshot, grants, FakeGenerator([reply]))
    assert not isinstance(off, Deferred), "the default must answer, as it did before M35"


def test_a_containing_phrase_is_why_this_cannot_be_fixed_by_widening():
    """The half of the 12 that looks like a matcher bug, and the reason it is not one.

    Beacon refused `total revenue` while `revenue` is certified -- a phrase CONTAINING a defined
    term. Discharging containment would clear several of the false deferrals, and it would also
    ground "revenue per customer", which is a DERIVATION over a certified term and is the exact
    shape M35 exists to refuse. The two are the same lexical relation and the guard cannot want
    one without the other. Widening is not the road back.
    """
    assert ungrounded_terms(["total revenue"], _defs()) == ["total revenue"]
    assert ungrounded_terms(["revenue per customer"], _defs()) == ["revenue per customer"]


def test_one_setting_reaches_both_ends_of_the_guard():
    """The flag crosses four hops -- Settings, `build_agent`, `Agent`, `plan_query` -- plus a fifth
    into the prompt, and every one of them defaults False. Dropping the kwarg at any hop leaves the
    guard inert with the whole suite green, because every other test passes it explicitly. That is
    the wired-at-one-end shape this branch produced three times: the guided schema not declaring
    the field, the prompt asking while the check was gated off, and the eval path's Agent never
    receiving the flag.

    Walked from `Settings` outward, in BOTH states, ending at the two things that actually
    consume the flag -- the prompt the model is sent, and the schema the request carries. A test
    that only checks the on-path cannot see a hop that ignores its argument, and a test that calls
    `system_prompt` directly is reading the call site rather than exercising it.

    `eval.engine.build_engine` builds its own `Agent` rather than going through `build_agent`, and
    this walk does not reach it -- its only test caller is skipped without a live key, and asserts
    nothing about the flag even when it runs. That hop is covered by
    `test_every_agent_construction_passes_the_flag` below, which is a source invariant rather than
    a behavioural walk: it is the only form that holds for a construction site nobody has written
    yet.
    """
    import duckdb

    from mnemiq.agent.modes import MODES, build_agent
    from mnemiq.assembly import build_components
    from mnemiq.config import Settings
    from mnemiq.generate.generator import _GUIDED_SQL_SCHEMA, _guided_extra_body

    assert Settings.model_fields["guard_undefined_terms"].default is False, (
        "the shipped default is off -- 12 false deferrals in 24, measured"
    )

    class _Adapter:
        dialect = "duckdb"

    for wanted in (True, False):
        # A base_url/api_key because `build_components` builds a real client; nothing calls it.
        settings = Settings(guard_undefined_terms=wanted,
                            llm_base_url="http://localhost:1/v1", llm_api_key="unused")

        agent = build_agent(
            MODES["thinking"], generator=None, synthesizer=None, adapter=None, cache=None,
            corrector=None, values=None, selector=None,
            guard_undefined_terms=settings.guard_undefined_terms,
        )
        assert agent.guard_undefined_terms is wanted, "Settings -> build_agent -> Agent"

        # The hop this is easiest to drop, because nothing downstream of it fails loudly: with the
        # prompt silent the model declares nothing, `ungrounded_terms([])` is `[]`, and the guard
        # is permanently inert with every test still green.
        con = duckdb.connect()  # ValueIndex reads a real connection at build time
        kit = build_components(settings, _Adapter(), con)
        assert kit.generator._declare_assumed_terms is wanted, (
            "Settings -> build_components -> LLMGenerator"
        )

        body = _guided_extra_body(True, declare_assumed_terms=wanted)
        properties = body["response_format"]["json_schema"]["schema"]["properties"]
        assert ("assumed_terms" in properties) is wanted, (
            "the SENT schema must declare the property only when the guard reads it"
        )
        assert "assumed_terms" in _GUIDED_SQL_SCHEMA["properties"], (
            "and the module constant must not be mutated by the strip"
        )


def test_every_agent_construction_passes_the_flag():
    """The invariant no per-call test can hold, because the failure is a call site that does not
    exist yet.

    `Agent` is constructed in three places -- `agent.modes.build_agent`, `eval.engine.build_engine`
    and `scripts/answer.py` -- and only the first has an offline test. The scan pins FOUR paths,
    because it covers both callables: the fourth is `runtime.py`, which calls `build_agent` rather
    than constructing an Agent, and has to pass the flag for the same reason. Beacon confirmed the second
    is uncovered on their side too: their in-process SUT injects an `engine_builder` and every
    beacon test of it passes a stub, so the real `build_engine` runs in no beacon test. `test_eval_live`
    does call it, but only with a live key configured and it asserts nothing about this flag. So
    deleting the kwarg leaves both suites green. Their seam is right for them -- beacon's CI has no
    mnemiq -- which means the guarantee has to live here.

    A source-level invariant instead, the shape this repo already uses for the ad-hoc env reads
    `test_no_adhoc_retrieval_k_reads` forbids. It costs nothing and it holds for the FOURTH site,
    which is the one that will actually cause this: three hops on this branch were left
    disconnected while their author was watching for exactly that failure.

    **The scan fails closed**, because a scanner that finds nothing is the same silent inertness
    one level up. It pins the exact set of files it expects to match -- not a count, which a site
    that stopped matching could still clear by being replaced with a new one; it counts `**kwargs`
    as not passing the flag, since a config-driven site is the likeliest next one and `Agent(**cfg)`
    proves nothing; and it resolves import aliases, so `Agent as _A` cannot rename its way out.

    Pinning the set means a legitimate new construction site fails this test. That is intended: it
    is a two-line edit here and a reminder to pass the flag, which is the whole point.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    roots = [root / "src" / "mnemiq", root / "scripts"]
    for d in roots:
        assert d.is_dir(), f"{d} must exist, or this scan guarantees nothing"

    found, missing = [], []
    for base in roots:
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text())
            # Local spellings of the two callables, including `from ... import Agent as _A`.
            names = {"Agent", "build_agent"}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in ("Agent", "build_agent") and alias.asname:
                            names.add(alias.asname)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name not in names:
                    continue
                where = f"{path.relative_to(root)}:{node.lineno} {name}(...)"
                found.append(where)
                # `**kwargs` does NOT count: it cannot be read here, and a config-driven site is
                # exactly where this goes wrong next.
                if "guard_undefined_terms" not in {kw.arg for kw in node.keywords}:
                    missing.append(where)

    # The exact set, not a floor. A floor of three still clears when one known site stops matching
    # -- rebound through an assignment, or moved to a directory outside these roots -- and the
    # dropped site goes unchecked while the test stays green.
    expected = {
        "src/mnemiq/runtime.py",
        "src/mnemiq/agent/modes.py",
        "src/mnemiq/eval/engine.py",
        "scripts/answer.py",
    }
    assert {f.split(":")[0] for f in found} == expected, (
        f"the set of Agent construction sites changed; the scan matched {sorted(found)}. If this "
        "is a new site, add it here AND pass guard_undefined_terms. If a known one vanished, the "
        "scan stopped seeing it and this invariant is no longer guarding it"
    )
    assert missing == [], (
        "every Agent/build_agent construction must pass guard_undefined_terms explicitly, or the "
        "guard is inert there with both suites green: " + "; ".join(missing)
    )
