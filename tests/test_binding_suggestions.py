import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.ontology.binder import (
    binding_suggestions_document,
    bind_schemes,
    suggest_bindings,
)
from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords


def _scheme(id_, label, notations):
    return ConceptScheme(id=id_, label=label, concepts=[
        Concept(id=f"{id_}:{n}", notation=n, pref_label=f"Label {n}") for n in notations])


def _snap(tmp_path, column, values):
    path = tmp_path / "s.sqlite"
    con = sqlite3.connect(path)
    rows = ",".join(f"({i + 1},'{v}')" for i, v in enumerate(values))
    con.executescript(
        f"CREATE TABLE sample (sample_pk INTEGER PRIMARY KEY, {column} TEXT);"
        f"INSERT INTO sample VALUES {rows};")
    con.commit()
    con.close()
    adapter = SQLiteAdapter(str(path))
    return adapter, enrich_structural(adapter, "p")


def test_near_miss_containment_is_suggested_not_bound(tmp_path):
    # 4 of 5 values match -> containment 0.80 (below strict 0.95); name "status" ~ "Status Code".
    adapter, snap = _snap(tmp_path, "status", ["a", "b", "c", "d", "zzz"])
    records = OntologyRecords(schemes=[_scheme("st", "Status Code", ["a", "b", "c", "d"])])
    bound = bind_schemes(adapter, snap, records)
    assert all(c.code_scheme is None for c in bound.columns)  # did NOT auto-bind
    suggestions = suggest_bindings(adapter, bound, records)
    hit = [s for s in suggestions if s.column_id == "sample.status"]
    assert len(hit) == 1
    assert hit[0].scheme_id == "st"
    assert hit[0].reason == "near_miss_containment"
    assert abs(hit[0].containment - 0.8) < 1e-9


def test_ambiguous_multi_scheme_is_suggested(tmp_path):
    # values are a clean subset of BOTH schemes, both label-affine -> bind nothing, suggest both.
    adapter, snap = _snap(tmp_path, "state", ["ca", "ny", "tx", "wa"])
    records = OntologyRecords(schemes=[
        _scheme("s1", "State Code", ["ca", "ny", "tx", "wa"]),
        _scheme("s2", "State Abbrev", ["ca", "ny", "tx", "wa"]),
    ])
    bound = bind_schemes(adapter, snap, records)
    assert all(c.code_scheme is None for c in bound.columns)  # ambiguity -> bound nothing
    suggestions = suggest_bindings(adapter, bound, records)
    hit = {s.scheme_id for s in suggestions if s.column_id == "sample.state"}
    assert hit == {"s1", "s2"}
    assert all(s.reason == "ambiguous" for s in suggestions if s.column_id == "sample.state")


def test_clean_auto_bind_is_excluded(tmp_path):
    adapter, snap = _snap(tmp_path, "status", ["a", "b", "c", "d", "e"])
    records = OntologyRecords(schemes=[_scheme("st", "Status Code", ["a", "b", "c", "d", "e"])])
    bound = bind_schemes(adapter, snap, records)
    assert any(c.code_scheme is not None for c in bound.columns)  # auto-bound
    suggestions = suggest_bindings(adapter, bound, records)
    assert not [s for s in suggestions if s.column_id == "sample.status"]


def test_no_signal_is_not_suggested(tmp_path):
    adapter, snap = _snap(tmp_path, "status", ["red", "green", "blue", "pink"])
    records = OntologyRecords(schemes=[_scheme("st", "Status Code", ["a", "b", "c", "d"])])
    bound = bind_schemes(adapter, snap, records)
    assert not suggest_bindings(adapter, bound, records)


def test_document_shape():
    doc = binding_suggestions_document("acme", [])
    assert doc["source_id"] == "acme"
    assert doc["suggestions"] == []
    assert "generated_at" in doc
