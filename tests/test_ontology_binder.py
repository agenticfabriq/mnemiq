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


def test_ontology_definitions_stay_unbound_and_therefore_visible(tmp_path):
    """Regression: binding a scheme definition to every table using the scheme was backwards.
    select_definitions requires ALL bound objects to be granted, so the more widely a scheme was
    used the FEWER identities could see what it means -- on Pagila the MPAA definition bound to
    film plus two views, and an analyst granted only `film` saw nothing. A scheme definition
    describes the scheme, not the tables, and its text names no table, so it discloses nothing.
    A proprietary taxonomy needs a public/proprietary marker in the format instead (SP2)."""
    from mnemiq.authz.grants import GrantSet
    from mnemiq.contract import Definition
    from mnemiq.ontology.binder import bind_schemes
    from mnemiq.semantic.glossary import select_definitions

    codes = ["R", "G", "B", "Y", "P"]
    adapter, snap = _snapshot(tmp_path, "defs", _table("colour_code", codes))
    records = OntologyRecords(
        schemes=[_scheme("urn:colour", "Colour Codes", codes)],
        definitions=[Definition(id="ontology:scheme:urn:colour", term="Colour Codes",
                                domain="ontology", definition="A vocabulary of colours.")],
    )
    out = bind_schemes(adapter, snap, records)
    assert next(d for d in out.definitions if d.term == "Colour Codes").bound_objects == []

    # and it survives grant filtering for an identity holding only one of the scheme's tables
    selected = select_definitions("what is the Colour Codes scheme", out.definitions,
                                  GrantSet(frozenset({"sample"})))
    assert [d.term for d in selected] == ["Colour Codes"]
