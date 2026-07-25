import mnemiq.contract  # noqa: F401  -- load contract first (pre-existing ontology.records import-order cycle)
from mnemiq.ontology.records import ConceptScheme, OntologyRecords, merge_records


def _scheme(id_, label):
    return ConceptScheme(id=id_, label=label, description=f"{label} desc", concepts=[])


def test_certified_scheme_wins_on_id_collision_and_unions_the_rest():
    local = OntologyRecords(schemes=[_scheme("mpaa", "local MPAA"), _scheme("local_only", "L")],
                            definitions=[])
    merged = merge_records(local, [_scheme("mpaa", "certified MPAA"), _scheme("cert_only", "C")])
    by_id = {s.id: s.label for s in merged.schemes}
    assert by_id["mpaa"] == "certified MPAA"          # certified wins on collision
    assert by_id["local_only"] == "L"                  # local-only kept
    assert by_id["cert_only"] == "C"                   # certified-only added


def test_merge_tolerates_no_local_records():
    merged = merge_records(None, [_scheme("mpaa", "MPAA")])
    assert [s.id for s in merged.schemes] == ["mpaa"]
