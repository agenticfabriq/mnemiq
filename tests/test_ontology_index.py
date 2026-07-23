import duckdb

from mnemiq.contract import CodeScheme, Column, Snapshot
from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords


def _records():
    return OntologyRecords(schemes=[
        ConceptScheme(id="urn:icd10", label="ICD-10-CM", concepts=[
            Concept(id="c1", notation="E11", pref_label="Type 2 diabetes mellitus",
                    alt_labels=["adult-onset diabetes"]),
            Concept(id="c2", notation="E10", pref_label="Type 1 diabetes mellitus"),
            Concept(id="c3", notation="I10", pref_label="Essential hypertension"),
        ]),
        ConceptScheme(id="urn:unused", label="Unused Scheme", concepts=[
            Concept(id="c9", notation="ZZ", pref_label="Never indexed")]),
    ])


def _snapshot():
    return Snapshot(version="v", source_id="s", created_at="t", columns=[
        Column(id="patient.icd10_cd", object_id="patient", name="icd10_cd",
               code_scheme=CodeScheme(id="urn:icd10", label="ICD-10-CM"))])


def test_build_indexes_only_bound_schemes():
    from mnemiq.semantic.ontology_index import build_ontology_index

    con = duckdb.connect(":memory:")
    n = build_ontology_index(_records(), _snapshot(), con)
    assert n == 4  # 3 prefLabels + 1 altLabel; the unused scheme is not indexed
    schemes = {
        r[0] for r in con.execute("SELECT DISTINCT scheme_id FROM ontology_concept").fetchall()
    }
    assert schemes == {"urn:icd10"}


def test_nearest_matches_labels_and_synonyms():
    from mnemiq.semantic.ontology_index import OntologyIndex, build_ontology_index

    con = duckdb.connect(":memory:")
    build_ontology_index(_records(), _snapshot(), con)
    index = OntologyIndex(con)

    assert index.nearest("urn:icd10", "type 2 diabetes")[0][0] == "E11"
    assert index.nearest("urn:icd10", "adult onset diabetes")[0][0] == "E11"
    assert index.nearest("urn:icd10", "xylophone repair", floor=0.5) == []


def test_nearest_returns_each_code_once():
    from mnemiq.semantic.ontology_index import OntologyIndex, build_ontology_index

    con = duckdb.connect(":memory:")
    build_ontology_index(_records(), _snapshot(), con)
    got = OntologyIndex(con).nearest("urn:icd10", "diabetes", floor=0.0)
    assert len(got) == len({notation for notation, _label in got})


def test_rebuild_replaces_rather_than_appends():
    from mnemiq.semantic.ontology_index import build_ontology_index

    con = duckdb.connect(":memory:")
    build_ontology_index(_records(), _snapshot(), con)
    build_ontology_index(_records(), _snapshot(), con)
    (total,) = con.execute("SELECT count(*) FROM ontology_concept").fetchone()
    assert total == 4


def test_unindexed_store_answers_cleanly():
    from mnemiq.semantic.ontology_index import OntologyIndex

    assert OntologyIndex(duckdb.connect(":memory:")).nearest("urn:icd10", "anything") == []
