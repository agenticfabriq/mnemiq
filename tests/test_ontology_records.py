import json


def test_load_records_parses_and_rejects_malformed(tmp_path):
    from mnemiq.ontology.records import load_records

    good = tmp_path / "r.json"
    good.write_text(json.dumps({
        "version": "v1",
        "schemes": [{
            "id": "urn:example:icd10", "label": "ICD-10-CM",
            "concepts": [{"id": "urn:example:icd10:E11", "notation": "E11",
                          "pref_label": "Type 2 diabetes mellitus",
                          "alt_labels": ["adult-onset diabetes"]}],
        }],
        "definitions": [],
        "bindings": {"patient.icd10_cd": "urn:example:icd10"},
    }))
    r = load_records(str(good))
    assert r.schemes[0].concepts[0].notation == "E11"
    assert r.schemes[0].concepts[0].alt_labels == ["adult-onset diabetes"]
    assert r.bindings["patient.icd10_cd"] == "urn:example:icd10"

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    try:
        load_records(str(bad))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_records_version_is_content_addressed_and_ignores_version_field():
    from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords, records_version

    def make(label, version):
        return OntologyRecords(
            version=version,
            schemes=[ConceptScheme(id="s", label=label, concepts=[
                Concept(id="c", notation="E11", pref_label="Type 2 diabetes mellitus")])],
        )

    assert records_version(make("ICD-10-CM", "a")) == records_version(make("ICD-10-CM", "b"))
    assert records_version(make("ICD-10-CM", "a")) != records_version(make("NAICS", "a"))
