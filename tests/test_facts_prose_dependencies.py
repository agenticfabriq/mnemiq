"""M143's prose: a GRAIN or GOTCHA line reaches only an identity that can see what it depends on.

The facts block is identity-independent text attached after the card is scoped, and fact
enrichment is given the table's relationships, so "join layoff_plans on secret_code" is a normal
gotcha. The owner's decisions (2026-10-14): enrichment records each fact's dependencies -- the
columns the model says it refers to, checked against the columns it was given, together with any
column or table its text names -- and a fact is withheld from an identity that cannot read one of
them. A fact with none recorded (enriched before this, or a certified record that never carried
them) is shown only to an identity that reads every column of its table raw and holds every
related table.

Dependencies are keyed by the fact's own TEXT, so a fact edited after enrichment -- a certified
override rewording a gotcha -- has none recorded and falls back to the strict rule.
"""

from __future__ import annotations

import json

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, JoinKey, Relationship, TableFacts
from mnemiq.enrichment.facts import parse_facts
from mnemiq.semantic.retrieval import RetrievedCard, _attach_facts

COLUMNS = [
    Column(id="orders.id", object_id="orders", name="id"),
    Column(id="orders.secret_code", object_id="orders", name="secret_code", pii_level="phi"),
    Column(id="orders.plan_id", object_id="orders", name="plan_id"),
    Column(id="layoff_plans.plan_id", object_id="layoff_plans", name="plan_id"),
    Column(id="layoff_plans.region", object_id="layoff_plans", name="region"),
]
RELS = [Relationship(id="o->l", **{"from": "orders"}, to="layoff_plans", cardinality="many_to_one",
                     join_keys=[JoinKey(left="plan_id", right="plan_id")])]
RELATED = {"layoff_plans": {"plan_id", "region"}}
OWN = {"id", "secret_code", "plan_id"}

EVERYTHING = GrantSet(frozenset({"orders", "layoff_plans"}), pii_clearance=frozenset({"phi"}))
NO_PHI = GrantSet(frozenset({"orders", "layoff_plans"}))
NO_PLANS = GrantSet(frozenset({"orders"}), pii_clearance=frozenset({"phi"}))


def _reply(**payload) -> str:
    return json.dumps(payload)


# -- recording, at enrichment -------------------------------------------------------------------

def test_a_gotcha_records_the_columns_the_model_names_and_the_ones_its_text_names():
    tf = parse_facts(_reply(gotchas=[
        {"text": "the diagnosis code is free text", "reads": ["secret_code"]},
        {"text": "join layoff_plans on plan_id", "reads": []},
    ]), "orders", OWN, {"plan_id", "region"}, related_tables=RELATED)
    assert tf.reads["the diagnosis code is free text"] == ["orders.secret_code"]
    # Declared nothing, and the scan still records what the sentence names.
    assert set(tf.reads["join layoff_plans on plan_id"]) >= {"layoff_plans", "orders.plan_id"}


def test_a_gotcha_declaring_a_column_it_was_not_given_has_none_recorded():
    """An invented name cannot be checked against any grant: unrecorded, so the strict rule."""
    tf = parse_facts(_reply(gotchas=[{"text": "watch the hidden flag", "reads": ["hidden_flag"]}]),
                     "orders", OWN, {"plan_id", "region"}, related_tables=RELATED)
    assert tf.gotchas == ["watch the hidden flag"] and "watch the hidden flag" not in tf.reads


def test_a_plain_string_gotcha_has_none_recorded():
    tf = parse_facts(_reply(gotchas=["amounts are text"], grain="one row per order",
                            grain_reads=["id"]),
                     "orders", OWN, {"plan_id", "region"}, related_tables=RELATED)
    assert "amounts are text" not in tf.reads
    assert tf.reads["one row per order"] == ["orders.id"]


def test_without_the_related_tables_nothing_is_recorded():
    """The old call shape cannot qualify a related column, so it records nothing at all."""
    tf = parse_facts(_reply(gotchas=[{"text": "x", "reads": ["id"]}]), "orders", OWN, set())
    assert tf.reads == {}


# -- withholding, per identity ------------------------------------------------------------------

def _block(tf: TableFacts, grants: GrantSet, relationships=RELS) -> str:
    card = RetrievedCard(object_id="orders", card="TABLE orders\n  id int", score=1.0)
    _attach_facts([card], [tf], grants=grants, columns=COLUMNS, relationships=relationships)
    return card.card


RECORDED = TableFacts(object_id="orders", grain="one row per order", gotchas=[
    "the diagnosis code is free text", "join layoff_plans on plan_id", "ids are dense"],
    reads={"one row per order": ["orders.id"],
           "the diagnosis code is free text": ["orders.secret_code"],
           "join layoff_plans on plan_id": ["layoff_plans", "orders.plan_id"],
           "ids are dense": []})


def test_a_fact_is_withheld_from_an_identity_that_cannot_read_what_it_depends_on():
    no_phi, no_plans = _block(RECORDED, NO_PHI), _block(RECORDED, NO_PLANS)
    assert "diagnosis" not in no_phi and "layoff_plans" in no_phi
    assert "layoff_plans" not in no_plans and "diagnosis" in no_plans
    for text in (no_phi, no_plans):
        assert "one row per order" in text and "ids are dense" in text


def test_everything_is_shown_to_an_identity_that_sees_everything():
    block = _block(RECORDED, EVERYTHING)
    assert all(g in block for g in RECORDED.gotchas) and "one row per order" in block


def test_an_unrecorded_fact_reaches_only_an_unrestricted_identity():
    legacy = TableFacts(object_id="orders", grain="one row per order",
                        gotchas=["the diagnosis code is free text"])
    assert "diagnosis" in _block(legacy, EVERYTHING)
    for grants in (NO_PHI, NO_PLANS):
        block = _block(legacy, grants)
        assert "diagnosis" not in block and "one row per order" not in block


def test_unknown_relationships_withhold_every_unrecorded_fact():
    """With no relationships to read, whether the identity holds every related table cannot be
    established: fail closed, even for the identity that sees everything."""
    legacy = TableFacts(object_id="orders", gotchas=["the diagnosis code is free text"])
    assert "diagnosis" not in _block(legacy, EVERYTHING, relationships=None)


def test_a_reworded_fact_loses_its_record():
    """A certified override that rewrites a gotcha keeps the old `reads` keyed by the old text."""
    edited = RECORDED.model_copy(update={"gotchas": ["ids are dense, mostly"]})
    assert "ids are dense, mostly" not in _block(edited, NO_PHI)
    assert "ids are dense, mostly" in _block(edited, EVERYTHING)


def test_a_dependency_on_an_object_the_snapshot_lacks_withholds_the_fact():
    tf = TableFacts(object_id="orders", gotchas=["see archive"], reads={"see archive": ["archive"]})
    assert "see archive" not in _block(tf, EVERYTHING)


# -- the join: what a deployment calls ----------------------------------------------------------

def test_retrieve_hands_the_snapshot_relationships_to_the_facts_block(tmp_path):
    from mnemiq.contract import IdentityContext, Snapshot
    from mnemiq.llm.embeddings import FakeEmbedder
    from mnemiq.semantic.retrieval import retrieve
    from mnemiq.semantic.store import build_index
    from mnemiq.store.bootstrap import init_store

    class _Authz:
        def __init__(self, grants):
            self.grants = grants

        def grants_for(self, _identity):
            return self.grants

    legacy = TableFacts(object_id="orders", gotchas=["the diagnosis code is free text"])
    snapshot = Snapshot(version="v1", source_id="s", created_at="t", columns=COLUMNS,
                        relationships=RELS, table_facts=[legacy])
    con = init_store(str(tmp_path / "s.duckdb"))
    embedder = FakeEmbedder()
    build_index(con, snapshot, embedder)
    identity = IdentityContext(tenant_id="t", principal_id="p", roles=["r"])

    def text(grants):
        packet = retrieve(con, "orders", identity, _Authz(grants), embedder, columns=COLUMNS,
                          table_facts=[legacy], snapshot=snapshot)
        return "\n".join(c.card for c in packet.cards if c.object_id == "orders")

    assert "diagnosis" in text(EVERYTHING), "the control: the block must arrive at all"
    assert "diagnosis" not in text(NO_PLANS)


def _wide_snapshot():
    """Two tables, every column described, a legacy gotcha on the wide one -- so a window too
    small for descriptions makes the fit re-render its cards and re-attach the facts."""
    from mnemiq.contract import Snapshot, SourceBinding

    columns = [Column(id=f"{t}.{n}", object_id=t, name=n, data_type="text",
                      description=f"The {n} of each {t}, described at some length for the fit.")
               for t, names in {"orders": ["plan_id"] + [f"c{i}" for i in range(30)],
                                "layoff_plans": ["plan_id", "region"]}.items() for n in names]
    return Snapshot(version="v1", source_id="s", created_at="t", columns=columns,
                    source_bindings=[SourceBinding(id=f"sb:{t}", source_id="s", object_id=t,
                                                   source_object=t, binding_type="table")
                                     for t in ("orders", "layoff_plans")],
                    relationships=RELS,
                    table_facts=[TableFacts(object_id="orders",
                                            gotchas=["the plan join drops cancelled orders"])])


def test_the_fit_re_render_withholds_as_retrieval_does():
    """The join for `semantic.fit`: past the window it rebuilds every card and re-attaches the
    facts, so it must hand over the snapshot's relationships too."""
    from mnemiq.llm import window
    from mnemiq.semantic.cards import build_cards
    from mnemiq.semantic.fit import FEEDBACK_ALLOWANCE, PromptFitter
    from mnemiq.semantic.retrieval import ContextPacket
    from mnemiq.sql.policy import build_access_policy

    snap = _wide_snapshot()
    tiny = 4000 + FEEDBACK_ALLOWANCE + 10

    def fitted(grants):
        cards = build_cards(snap, policy=build_access_policy(snap, grants))
        packet = ContextPacket(question="orders per plan", grant_fingerprint="fp",
                               enrichment_version=None,
                               cards=[RetrievedCard(object_id=c.object_id, card=c.text, score=1.0)
                                      for c in cards])
        fitter = PromptFitter(dialect="duckdb", card_style="cards", reply_tokens=4000, window=tiny,
                              count=lambda s, u: (len(s + u) // 4, tiny))
        out = fitter.fit(packet, snap, grants)
        assert out.descriptions is not None, "the re-render has to run, or this proves nothing"
        return "\n".join(c.card for c in out.cards)

    assert "cancelled orders" in fitted(window.everything(snap))
    assert "cancelled orders" not in fitted(GrantSet(frozenset({"orders"})))


def test_the_window_check_measures_only_what_the_role_would_be_sent():
    """The join for `llm.window`: the boot check sends each role's largest prompt to the model
    server to count, and nothing sent there may exceed what that role could be sent."""
    import duckdb

    from mnemiq.config import Settings
    from mnemiq.llm import window

    snap = _wide_snapshot()
    settings = Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m")

    def real(grants):
        return window.largest_prompts(duckdb.connect(), snap, settings, "duckdb", grants)[2]

    assert "cancelled orders" in real(window.everything(snap))
    assert "cancelled orders" not in real(GrantSet(frozenset({"orders"})))


def test_a_grain_and_a_gotcha_with_the_same_words_share_one_record():
    """Review gate: keyed by text, the grain's list overwrote the gotcha's, so a gotcha declaring
    `secret_code` reached an identity that cannot read it whenever the grain said the same."""
    same = "codes repeat across orders"
    tf = parse_facts(_reply(grain=same, grain_reads=[],
                            gotchas=[{"text": same, "reads": ["secret_code"]}]),
                     "orders", OWN, {"plan_id", "region"}, related_tables=RELATED)
    assert "orders.secret_code" in tf.reads[same]
    unrecorded = parse_facts(_reply(grain=same, gotchas=[{"text": same, "reads": ["id"]}]),
                             "orders", OWN, {"plan_id", "region"}, related_tables=RELATED)
    assert same not in unrecorded.reads, "one occurrence with none recorded leaves none"


def _federated():
    from mnemiq.config import SourceSpec
    from mnemiq.contract import Snapshot
    from mnemiq.semantic.federation import merge_snapshots

    snap = Snapshot(version="v1", source_id="s", created_at="t", columns=COLUMNS,
                    relationships=RELS, table_facts=[RECORDED,
                                                     TableFacts(object_id="layoff_plans",
                                                                gotchas=["plans are regional"])])
    return merge_snapshots([(SourceSpec(id="s", kind="postgres", target="x", catalog="pg",
                                        schema="public"), snap)])


def test_a_federated_snapshot_resolves_the_dependencies_it_qualified():
    """Review gate, the blocker: the merge qualified each fact's table and not its `reads`, so
    no dependency resolved and every recorded fact was withheld from everyone."""
    from mnemiq.llm import window
    from mnemiq.semantic.retrieval import known_relationships

    merged = _federated()
    (orders,) = [tf for tf in merged.table_facts if tf.object_id == "pg.orders"]
    assert orders.reads["the diagnosis code is free text"] == ["pg.orders.secret_code"]

    def block(grants):
        card = RetrievedCard(object_id="pg.orders", card="TABLE pg.orders\n  id int", score=1.0)
        _attach_facts([card], merged.table_facts, grants=grants, columns=merged.columns,
                      relationships=known_relationships(merged))
        return card.card

    assert "diagnosis" in block(window.everything(merged))
    assert "diagnosis" not in block(GrantSet(frozenset({"pg.orders", "pg.layoff_plans"})))


def test_a_federated_snapshot_counts_its_relationships_as_unknown():
    """The merge does not carry relationships, so its empty list is not "no related tables": a
    fact with none recorded is withheld there, from everyone."""
    from mnemiq.llm import window
    from mnemiq.semantic.retrieval import known_relationships

    merged = _federated()
    assert known_relationships(merged) is None
    card = RetrievedCard(object_id="pg.layoff_plans", card="TABLE pg.layoff_plans", score=1.0)
    _attach_facts([card], merged.table_facts, grants=window.everything(merged),
                  columns=merged.columns, relationships=known_relationships(merged))
    assert "regional" not in card.card
