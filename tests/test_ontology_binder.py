import sqlite3

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords


def _scheme(id_, label, notations):
    return ConceptScheme(id=id_, label=label, concepts=[
        Concept(id=f"{id_}:{n}", notation=n, pref_label=f"Label {n}") for n in notations])


def _snapshot(tmp_path, name, ddl):
    path = tmp_path / f"{name}.sqlite"
    con = sqlite3.connect(path)
    con.executescript(ddl)
    con.commit()
    con.close()
    adapter = SQLiteAdapter(str(path))
    return adapter, enrich_structural(adapter, "p")


def _rows(values):
    return ",".join(f"({i + 1},'{v}')" for i, v in enumerate(values))


def _table(column, values):
    return f"""
        CREATE TABLE sample (sample_pk INTEGER PRIMARY KEY, {column} TEXT);
        INSERT INTO sample VALUES {_rows(values)};
    """


def test_auto_binds_when_values_are_contained_and_the_name_matches(tmp_path):
    from mnemiq.ontology.binder import bind_schemes

    codes = ["R", "G", "B", "Y", "P"]
    adapter, snap = _snapshot(tmp_path, "a", _table("colour_code", codes))
    records = OntologyRecords(schemes=[_scheme("urn:colour", "Colour Codes", codes)])

    out = bind_schemes(adapter, snap, records)
    col = next(c for c in out.columns if c.name == "colour_code")
    assert col.code_scheme is not None
    assert col.code_scheme.label == "Colour Codes"


def test_abbreviated_punctuated_scheme_name_still_binds(tmp_path):
    """Regression: `icd10_cd` vs `ICD-10-CM` scores 0.188 on raw trigrams -- below the gate.
    The canonical motivating case must not need an explicit binding to work."""
    from mnemiq.ontology.binder import bind_schemes

    codes = ["E10", "E11", "I10", "J45", "K21"]
    adapter, snap = _snapshot(tmp_path, "icd", _table("icd10_cd", codes))
    records = OntologyRecords(schemes=[_scheme("urn:icd10", "ICD-10-CM", codes)])

    out = bind_schemes(adapter, snap, records)
    assert next(c for c in out.columns if c.name == "icd10_cd").code_scheme is not None


def test_sentinels_are_forgiven_but_a_foreign_code_is_not(tmp_path):
    from mnemiq.ontology.binder import bind_schemes

    scheme = _scheme("urn:colour", "Colour Codes", ["R", "G", "B", "Y", "P"])

    adapter, snap = _snapshot(
        tmp_path, "dirty", _table("colour_code", ["R", "G", "B", "Y", "P", "UNKNOWN", "N/A"]))
    out = bind_schemes(adapter, snap, OntologyRecords(schemes=[scheme]))
    assert next(c for c in out.columns if c.name == "colour_code").code_scheme is not None

    adapter2, snap2 = _snapshot(
        tmp_path, "foreign", _table("colour_code", ["R", "G", "B", "Y", "ZZ1", "ZZ2", "ZZ3"]))
    out2 = bind_schemes(adapter2, snap2, OntologyRecords(schemes=[scheme]))
    assert next(c for c in out2.columns if c.name == "colour_code").code_scheme is None


def test_min_distinct_counts_non_sentinel_values_only(tmp_path):
    from mnemiq.ontology.binder import bind_schemes

    # sentinel padding must not carry a column over the minimum on two real codes
    adapter, snap = _snapshot(
        tmp_path, "padded",
        _table("colour_code", ["UNKNOWN", "N/A", "NULL", "-1", "?", "R", "G"]))
    records = OntologyRecords(schemes=[_scheme("urn:colour", "Colour Codes", ["R", "G", "B"])])
    out = bind_schemes(adapter, snap, records)
    assert next(c for c in out.columns if c.name == "colour_code").code_scheme is None


def test_name_affinity_is_required(tmp_path):
    from mnemiq.ontology.binder import bind_schemes

    codes = ["R", "G", "B", "Y", "P"]
    adapter, snap = _snapshot(tmp_path, "noaffinity", _table("zzz", codes))
    records = OntologyRecords(schemes=[_scheme("urn:colour", "Colour Codes", codes)])
    out = bind_schemes(adapter, snap, records)
    assert next(c for c in out.columns if c.name == "zzz").code_scheme is None


def test_two_matching_schemes_bind_nothing(tmp_path):
    from mnemiq.ontology.binder import bind_schemes

    codes = ["R", "G", "B", "Y", "P"]
    adapter, snap = _snapshot(tmp_path, "ambiguous", _table("colour_code", codes))
    records = OntologyRecords(schemes=[
        _scheme("urn:colour:a", "Colour Codes", codes),
        _scheme("urn:colour:b", "Colour Code List", codes),
    ])
    out = bind_schemes(adapter, snap, records)
    assert next(c for c in out.columns if c.name == "colour_code").code_scheme is None


def test_explicit_binding_bypasses_every_gate(tmp_path):
    from mnemiq.ontology.binder import bind_schemes

    adapter, snap = _snapshot(tmp_path, "explicit", _table("zzz", ["Q1", "Q2"]))
    records = OntologyRecords(
        schemes=[_scheme("urn:colour", "Colour Codes", ["R", "G", "B"])],
        bindings={"sample.zzz": "urn:colour"},
    )
    out = bind_schemes(adapter, snap, records)
    assert next(c for c in out.columns if c.name == "zzz").code_scheme.id == "urn:colour"


def test_bound_definitions_are_narrowed_to_the_bound_table(tmp_path):
    from mnemiq.contract import Definition
    from mnemiq.ontology.binder import bind_schemes

    codes = ["R", "G", "B", "Y", "P"]
    adapter, snap = _snapshot(tmp_path, "defs", _table("colour_code", codes))
    records = OntologyRecords(
        schemes=[_scheme("urn:colour", "Colour Codes", codes)],
        definitions=[Definition(id="ontology:scheme:urn:colour", term="Colour Codes",
                                domain="ontology", definition="A vocabulary of colours."),
                     Definition(id="ontology:term:urn:other", term="Sample Batch",
                                domain="ontology", definition="A group of samples.")],
    )
    out = bind_schemes(adapter, snap, records)
    by_term = {d.term: d for d in out.definitions}
    assert by_term["Colour Codes"].bound_objects == ["sample"]  # grant filtering now applies
    assert by_term["Sample Batch"].bound_objects == []          # unbound stays global
