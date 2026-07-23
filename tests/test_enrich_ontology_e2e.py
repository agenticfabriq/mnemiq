import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.grounding import ground_codes
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.ontology.digest import digest_ontology
from mnemiq.semantic.cards import build_cards


def test_ttl_to_grounded_card(tmp_path):
    """The whole static path: TTL -> records -> bind -> ground -> card."""
    path = tmp_path / "p.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sample (sample_pk INTEGER PRIMARY KEY, colour_code TEXT);
        INSERT INTO sample VALUES (1,'R'),(2,'G'),(3,'B'),(4,'Y'),(5,'P'),
                                  (6,'R'),(7,'G'),(8,'B');
    """)
    con.commit()
    con.close()

    adapter = SQLiteAdapter(str(path))
    snapshot = enrich_structural(adapter, "p")
    records = digest_ontology(["tests/fixtures/ontology/skos_scheme.ttl"])

    grounded = ground_codes(adapter, snapshot, ontology=records)
    column = next(c for c in grounded.columns if c.name == "colour_code")

    assert column.code_scheme.label == "Colour Codes"
    assert {cv.code: cv.meaning for cv in column.coded_values} == {
        "R": "Red", "G": "Green", "B": "Blue", "Y": "Yellow", "P": "Purple"}
    assert all(cv.source == "ontology" for cv in column.coded_values)

    card = next(c for c in build_cards(grounded) if c.object_id == "sample").text
    assert "R = Red" in card
    assert "Colour Codes" in card


def test_ttl_to_question_time_candidates(tmp_path):
    """The whole runtime path for a LARGE code system: nothing is harvested into the snapshot,
    so the concept index is what carries the meaning to the prompt."""
    import duckdb

    from mnemiq.generate.prompts import user_prompt
    from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords
    from mnemiq.semantic.ontology_index import OntologyIndex, build_ontology_index
    from mnemiq.semantic.retrieval import ContextPacket, _resolve_concepts

    codes = [("E10", "Type 1 diabetes mellitus"), ("E11", "Type 2 diabetes mellitus"),
             ("I10", "Essential hypertension"), ("J45", "Asthma"), ("K21", "Reflux disease")]
    path = tmp_path / "clinic.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE patient (patient_pk INTEGER PRIMARY KEY, icd10_cd TEXT);"
        + "INSERT INTO patient VALUES "
        + ",".join(f"({i + 1},'{c}')" for i, (c, _) in enumerate(codes)) + ";"
    )
    con.commit()
    con.close()

    adapter = SQLiteAdapter(str(path))
    snapshot = enrich_structural(adapter, "clinic")
    records = OntologyRecords(schemes=[ConceptScheme(
        id="urn:icd10", label="ICD-10-CM",
        concepts=[Concept(id=f"c{c}", notation=c, pref_label=lbl) for c, lbl in codes])])

    grounded = ground_codes(adapter, snapshot, ontology=records)
    column = next(c for c in grounded.columns if c.name == "icd10_cd")
    assert column.code_scheme.label == "ICD-10-CM"  # bound despite the punctuated scheme name

    store = duckdb.connect(":memory:")
    build_ontology_index(records, grounded, store)
    question = "how many patients have type 2 diabetes"
    concepts = _resolve_concepts(question, grounded.columns, {"patient"}, OntologyIndex(store))
    assert concepts[0].notation == "E11"

    packet = ContextPacket(question=question, cards=[], grant_fingerprint="f",
                           enrichment_version="v")
    packet.concepts = concepts
    text = user_prompt(packet)
    assert "CODE VOCABULARY" in text
    assert "E11 = Type 2 diabetes mellitus" in text
