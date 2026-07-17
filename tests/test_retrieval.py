from mnemiq.authz.grants import DenyAll, GrantSet
from mnemiq.contract import Column, Definition, IdentityContext, Snapshot, SourceBinding
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_index
from mnemiq.store.bootstrap import init_store


class _StaticAuthz:
    def __init__(self, *objects):
        self._grants = GrantSet(frozenset(objects))

    def grants_for(self, identity):
        return self._grants


_TABLES = {
    "claim": "claim_identifier",
    "policy": "policy_identifier",
    "layoff_plan": "headcount_reduction",
}


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        source_bindings=[
            SourceBinding(
                id=f"sb:{t}", source_id="acme", object_id=t, source_object=t, binding_type="table"
            )
            for t in _TABLES
        ],
        columns=[
            Column(id=f"{t}.{c}", object_id=t, name=c, data_type="text")
            for t, c in _TABLES.items()
        ],
    )


def _con(tmp_path):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, _snapshot(), FakeEmbedder())
    return con


def _identity():
    return IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def test_retrieves_the_lexically_matching_card(tmp_path):
    packet = retrieve(
        _con(tmp_path),
        "claim_identifier",
        _identity(),
        _StaticAuthz("claim", "policy"),
        FakeEmbedder(),
    )
    assert packet.cards[0].object_id == "claim"
    assert "claim_identifier" in packet.cards[0].card


def test_an_unauthorized_table_is_invisible_even_when_it_is_the_best_match(tmp_path):
    # the question is *about* the forbidden table -- it would win every ranking
    packet = retrieve(
        _con(tmp_path),
        "headcount_reduction layoff_plan",
        _identity(),
        _StaticAuthz("claim", "policy"),
        FakeEmbedder(),
    )
    ids = {c.object_id for c in packet.cards}
    assert "layoff_plan" not in ids

    # and nothing about it leaks -- not the name, not its existence
    blob = " ".join(c.card for c in packet.cards)
    assert "layoff" not in blob and "headcount" not in blob


def test_no_grants_means_an_empty_packet(tmp_path):
    packet = retrieve(_con(tmp_path), "claim", _identity(), DenyAll(), FakeEmbedder())
    assert packet.cards == []


def test_packet_carries_the_cache_key_material(tmp_path):
    authz = _StaticAuthz("claim", "policy")
    packet = retrieve(_con(tmp_path), "claim", _identity(), authz, FakeEmbedder())

    assert packet.question == "claim"
    assert packet.grant_fingerprint == authz.grants_for(_identity()).fingerprint
    assert packet.enrichment_version == "v1"
    assert packet.definitions == [] and packet.examples == []  # Plans 06/08 fill these


def test_k_bounds_the_packet(tmp_path):
    packet = retrieve(
        _con(tmp_path),
        "identifier",
        _identity(),
        _StaticAuthz("claim", "policy", "layoff_plan"),
        FakeEmbedder(),
        k=1,
    )
    assert len(packet.cards) == 1


def test_retrieve_fills_definitions_access_scoped(tmp_path):
    definition = Definition(
        id="def-claim", term="claim", domain="claims",
        definition="A demand for payment under a policy.", bound_objects=["claim"],
    )
    con = _con(tmp_path)

    packet = retrieve(
        con, "how many claims?", _identity(), _StaticAuthz("claim", "policy"),
        FakeEmbedder(), definitions=[definition],
    )
    assert packet.definitions == [definition]

    packet = retrieve(
        con, "how many claims?", _identity(), _StaticAuthz("policy"),
        FakeEmbedder(), definitions=[definition],
    )
    assert packet.definitions == []  # binds an ungranted table: never shown


def test_no_grants_means_no_definitions_either(tmp_path):
    definition = Definition(
        id="def-claim", term="claim", domain="claims",
        definition="A demand for payment under a policy.", bound_objects=["claim"],
    )
    packet = retrieve(
        _con(tmp_path), "how many claims?", _identity(), DenyAll(),
        FakeEmbedder(), definitions=[definition],
    )
    assert packet.definitions == []


def test_attach_facts_inserts_block_after_table_line():
    from mnemiq.contract import TableFacts
    from mnemiq.semantic.retrieval import RetrievedCard, _attach_facts

    card = RetrievedCard(object_id="claim", card="TABLE claim\nCOLUMNS:\n- amount", score=1.0)
    _attach_facts([card], [TableFacts(object_id="claim", grain="one row per claim")])
    assert card.card.startswith("TABLE claim\nGRAIN: one row per claim\nCOLUMNS:")
    # a table with no facts is untouched
    other = RetrievedCard(object_id="party", card="TABLE party\nCOLUMNS:", score=1.0)
    _attach_facts([other], [TableFacts(object_id="claim", grain="x")])
    assert other.card == "TABLE party\nCOLUMNS:"


def test_retrieve_examples_by_question_similarity_and_access_scope(tmp_path):
    from mnemiq.contract import Column, Example, Snapshot
    from mnemiq.llm.embeddings import FakeEmbedder
    from mnemiq.semantic.retrieval import _retrieve_examples
    from mnemiq.semantic.store import build_example_index
    from mnemiq.store.bootstrap import init_store

    con = init_store(str(tmp_path / "s.duckdb"))
    snap = Snapshot(
        version="v1", source_id="acme", created_at="t",
        columns=[Column(id="claim.n", object_id="claim", name="n")],
        examples=[
            Example(question="how many claims?", sql="SELECT count(*) FROM claim",
                    tables=["claim"], object_id="claim"),
            Example(question="party secret", sql="SELECT * FROM party", tables=["party"],
                    object_id="party"),  # outside the grant -> must never surface
        ],
    )
    emb = FakeEmbedder()
    build_example_index(con, snap, emb)
    (qvec,) = emb.embed(["how many claims?"])
    got = _retrieve_examples(con, qvec, allowed={"claim"}, k=5)
    assert [e.object_id for e in got] == ["claim"]


def test_retrieve_examples_no_index_returns_empty(tmp_path):
    from mnemiq.llm.embeddings import FakeEmbedder
    from mnemiq.semantic.retrieval import _retrieve_examples
    from mnemiq.store.bootstrap import init_store

    con = init_store(str(tmp_path / "s.duckdb"))  # no example table built
    (qvec,) = FakeEmbedder().embed(["q"])
    assert _retrieve_examples(con, qvec, allowed={"claim"}, k=5) == []
