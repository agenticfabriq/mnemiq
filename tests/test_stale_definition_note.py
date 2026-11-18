"""A definition that names codes its column holds none of is shown with a note saying so.

The writer is told definitions are authoritative and to follow them exactly. One written for other data names codes the
source does not hold, so the writer filters on them and the value check refuses the statement; told the real value, the
writer fixed it and then put the definition's code back on its next attempt, because the definition still said it.
`stale_codes` finds these definitions at load, for the warning; the note carries the same finding to every model
that reads the definition (`meaning_sections`).

Shown only to an identity granted every table that holds the column and reading it raw, as the finding is about all of
them, and only while the definition reads as it did when checked: one corrected since carries no note. The note names
the codes, which the definition already shows, and never a real value.
"""
from __future__ import annotations

import duckdb

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Definition, IdentityContext, Snapshot
from mnemiq.generate.prompts import meaning_sections
from mnemiq.llm.embeddings import FakeEmbedder
from mnemiq.semantic.definition_codes import noted_stale, stale_codes
from mnemiq.semantic.retrieval import ContextPacket, retrieve
from mnemiq.semantic.store import build_index
from mnemiq.semantic.values import ValueIndex
from mnemiq.store.bootstrap import init_store

STALE_TEXT = "An order ships from plant AB12: filter PLANT_CODE = 'AB12'."
STALE = Definition(id="d:plant", term="plant", domain="plant", definition=STALE_TEXT, bound_objects=["orders"])
FRESH = Definition(id="d:plant", term="plant", domain="plant", bound_objects=["orders"],
                   definition="An order ships from plant XY98: filter PLANT_CODE = 'XY98'.")


def _snapshot(*definitions: Definition, pii: str | None = None, tables=("orders",)) -> Snapshot:
    return Snapshot(version="v1", source_id="src", created_at="2026-11-18T00:00:00Z",
                    columns=[Column(id=f"{t}.PLANT_CODE", object_id=t, name="PLANT_CODE", data_type="VARCHAR",
                                    pii_level=pii) for t in tables],
                    definitions=list(definitions))


def _index(con, tables=("orders",)) -> ValueIndex:
    index = ValueIndex(con)
    for table in tables:
        index._con.execute("INSERT INTO value_index VALUES ('src', ?, 'PLANT_CODE', 'XY98')", [table])
    return index


def _packet(*definitions: Definition, stale=()) -> ContextPacket:
    return ContextPacket(question="q", cards=[], grant_fingerprint="fp", enrichment_version=None,
                         definitions=list(definitions), stale_codes=list(stale))


def test_a_report_carries_the_definition_it_read():
    (found,) = stale_codes(_snapshot(STALE), _index(duckdb.connect()))
    assert (found.term, found.definition, found.codes) == ("plant", STALE_TEXT, ("AB12",))


def test_the_definitions_given_are_the_ones_read():
    snap = _snapshot(FRESH)
    assert stale_codes(snap, _index(duckdb.connect())) == []
    assert [f.definition for f in stale_codes(snap, _index(duckdb.connect()), definitions=[STALE])] == [STALE_TEXT]


def test_the_note_follows_its_definition_and_names_no_real_value():
    found = stale_codes(_snapshot(STALE), _index(duckdb.connect()))
    lines = meaning_sections(_packet(STALE, stale=found))
    at = next(i for i, line in enumerate(lines) if line.startswith("- plant: "))
    note = lines[at + 1]
    assert note.startswith("  Note:") and "PLANT_CODE" in note and "'AB12'" in note
    # Never "drop the filter": the constraint stands, on the column's code of the same meaning, or the question is
    # declined (Codex on the note).
    assert "same meaning" in note and "rather than guess" in note and "do not filter" not in note
    assert "XY98" not in "\n".join(lines)


def test_a_definition_corrected_since_it_was_checked_has_no_note():
    found = stale_codes(_snapshot(STALE), _index(duckdb.connect()))
    lines = meaning_sections(_packet(FRESH, stale=found))
    assert not any("Note:" in line for line in lines)


def _retrieved(snap: Snapshot, grants: GrantSet, tmp_path, index_tables=("orders",)):
    con = init_store(str(tmp_path / "s.duckdb"))
    build_index(con, snap, FakeEmbedder())
    found = stale_codes(snap, _index(con, index_tables))

    class Authz:
        def grants_for(self, identity):
            return grants

    identity = IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])
    return retrieve(con, "orders by plant", identity, Authz(), FakeEmbedder(), definitions=snap.definitions,
                    columns=snap.columns, snapshot=snap, grants=grants, stale=found)


def test_retrieval_carries_the_note_of_a_definition_it_selects(tmp_path):
    packet = _retrieved(_snapshot(STALE), GrantSet(frozenset({"orders"})), tmp_path)
    assert packet.definitions == [STALE]
    assert [s.codes for s in packet.stale_codes] == [("AB12",)]
    assert any(line.startswith("  Note:") for line in meaning_sections(packet))


def test_no_note_where_the_identity_is_not_granted_every_table_holding_the_column(tmp_path):
    # Both tables hold PLANT_CODE, so "the data holds none of these codes" is about both; this identity reads one.
    snap = _snapshot(STALE, tables=("orders", "returns"))
    packet = _retrieved(snap, GrantSet(frozenset({"orders"})), tmp_path, index_tables=("orders", "returns"))
    assert packet.definitions == [STALE] and packet.stale_codes == []


def test_no_note_where_the_identity_does_not_read_the_column_raw(tmp_path):
    snap = _snapshot(STALE, pii="pii")
    packet = _retrieved(snap, GrantSet(frozenset({"orders"})), tmp_path)
    assert packet.definitions == [STALE] and packet.stale_codes == []
    (tmp_path / "c").mkdir()
    cleared = _retrieved(snap, GrantSet(frozenset({"orders"}), pii_clearance=frozenset({"pii"})), tmp_path / "c")
    assert [s.codes for s in cleared.stale_codes] == [("AB12",)]


def test_no_note_for_a_definition_retrieval_did_not_select():
    found = stale_codes(_snapshot(STALE), _index(duckdb.connect()))
    grants = GrantSet(frozenset({"orders"}))
    columns = _snapshot(STALE).columns
    assert noted_stale(found, [], grants, columns) == []
    assert noted_stale(found, [STALE], grants, columns) == found


# Both doors: the product computes the finding at boot and hands it to retrieval on every question, and the eval
# engine hands the same, so an eval measures the prompt `ask` sends (the two-doors test names `stale` too).
WIDE_TEXT = "A row of plant AB12: filter col_0 = 'AB12'."


def test_the_product_boots_with_the_finding_and_its_writer_reads_the_note(tmp_path, monkeypatch):
    import json
    import os

    import mnemiq.runtime as runtime_module
    from mnemiq.runtime import build_runtime
    from mnemiq.semantic.retrieval import RetrievedCard
    from mnemiq.store.snapshot_store import save_snapshot
    from tests.scripted_deployment import IDENTITY, SQL, deployment
    from tests.sweep.stub_model import StubModel

    handed = {}

    def one_table(con, question, identity, authz, embedder, **kwargs):
        handed.setdefault("first", kwargs.get("stale"))
        handed["stale"] = kwargs.get("stale")
        definitions = list(kwargs["definitions"])
        return ContextPacket(question=question, cards=[RetrievedCard(object_id="wide", card="wide (col_0 TEXT)",
                                                                     score=1.0)],
                             grant_fingerprint="fp", enrichment_version=None, definitions=definitions,
                             stale_codes=noted_stale(kwargs["stale"], definitions, kwargs["grants"],
                                                     kwargs["columns"]))

    monkeypatch.setattr(runtime_module, "retrieve", one_table)
    with StubModel(sql=SQL) as stub:
        settings = deployment(tmp_path, stub.url)
        # The deployment's store again, its snapshot holding a definition whose code col_0 does not hold.
        os.remove(settings.store_path)
        con = init_store(settings.store_path)
        save_snapshot(con, Snapshot(version="v1", source_id="only", created_at="t",
                                    columns=[Column(id="wide.col_0", object_id="wide", name="col_0",
                                                    data_type="VARCHAR")],
                                    definitions=[Definition(id="d:plant", term="plant", domain="plant",
                                                            definition=WIDE_TEXT, bound_objects=["wide"])]))
        ValueIndex(con)._con.execute("INSERT INTO value_index VALUES ('only', 'wide', 'col_0', 'XY98')")
        con.close()
        runtime = build_runtime(settings)
        assert runtime.findings.snapshot is runtime.snapshot
        assert [s.codes for s in runtime.findings.stale] == [("AB12",)]
        runtime.ask("rows of plant AB12", IDENTITY)
        # Findings made for another snapshot (a swap's, published before or after this one) reach no question.
        found = runtime.findings
        runtime.findings = type(found)(snapshot=runtime.snapshot.model_copy(), stale=found.stale)
        runtime.ask("rows of plant AB12", IDENTITY)
    assert handed["stale"] == ()
    assert handed["first"] == found.stale
    sent = [json.dumps(request) for request in stub.requests]
    assert any("Note:" in body and "'AB12'" in body and "XY98" not in body for body in sent)


def test_the_eval_engine_hands_retrieval_the_same_finding(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import mnemiq.eval.engine as engine
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.config import Settings

    handed = {}
    monkeypatch.setattr(engine, "LLMEmbedder", lambda settings: None)
    monkeypatch.setattr(engine, "build_index", lambda con, *a, **k: con.execute(
        "CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT)"))
    monkeypatch.setattr(engine, "build_example_index", lambda *a, **k: None)
    monkeypatch.setattr(engine, "build_value_index", lambda *a, **k: 0)

    def retrieve(con, question, identity, authz, embedder, **kw):
        handed["stale"] = kw.get("stale")
        return SimpleNamespace()

    monkeypatch.setattr(engine, "retrieve", retrieve)
    values = _index(duckdb.connect())

    class Fitter:
        def fit(self, packet, snapshot, grants):
            return packet

    monkeypatch.setattr(engine, "build_components", lambda settings, adapter, con: SimpleNamespace(
        fitter=Fitter(), client=None, generator=None, synthesizer=None, corrector=None, values=values,
        selector=None))

    class Agent:
        def __init__(self, **kw):
            pass

        def decline_if_too_long(self, question):
            return None

        def answer(self, packet, snapshot, grants, identity):
            return AgentAnswer(answer="A")

    monkeypatch.setattr(engine, "Agent", Agent)
    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
                        store_path=str(tmp_path / "s.duckdb"))
    ask, _ = engine.build_engine(_snapshot(STALE), adapter=None, settings=settings,
                                 grants=GrantSet(frozenset({"orders"})))
    ask("orders by plant")
    assert [(s.term, s.codes) for s in handed["stale"]] == [("plant", ("AB12",))]


def test_a_swap_during_the_answer_leaves_its_record_on_the_snapshot_it_read(tmp_path, monkeypatch):
    """`ask` reads the snapshot once; the trace's certified references come from that one, not from a snapshot swapped
    in while the answer ran, which would name another version's hash under the same ids (Codex's second pass)."""
    from types import SimpleNamespace

    import mnemiq.runtime as runtime_module
    from mnemiq.contract.semantic import CertifiedRef
    from mnemiq.runtime import build_runtime
    from mnemiq.semantic.retrieval import RetrievedCard
    from tests.scripted_deployment import IDENTITY, SQL, deployment
    from tests.sweep.stub_model import StubModel

    def ref(version):
        return [CertifiedRef(object_type="definition", object_id="d:plant", version_hash=version)]

    with StubModel(sql=SQL) as stub:
        runtime = build_runtime(deployment(tmp_path, stub.url))
        runtime.snapshot = runtime.snapshot.model_copy(update={"certified_refs": ref("A")})
        newer = runtime.snapshot.model_copy(update={"certified_refs": ref("B")})

        def swapped_while_retrieving(con, question, identity, authz, embedder, **kwargs):
            runtime.snapshot = newer
            return ContextPacket(question=question, cards=[RetrievedCard(object_id="wide", card="wide (col_0 TEXT)",
                                                                         score=1.0)],
                                 grant_fingerprint="fp", enrichment_version=None)

        monkeypatch.setattr(runtime_module, "retrieve", swapped_while_retrieving)
        events = []
        runtime.trace_sink = SimpleNamespace(record_answer=events.append)
        runtime.ask("q", IDENTITY)
    assert [r.version_hash for r in events[0].certified_refs] == ["A"]
