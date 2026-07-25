"""A changed governed ontology (a Verity-certified concept scheme) must invalidate the cached
enrichment, even when the local ontology_records_path is unchanged AND the column binding
(scheme id/label) is unchanged.

The risk: `Column.code_scheme` is only `{id, label}` -- it carries none of a scheme's concept
content (notations/pref_labels/definitions). So a certified scheme that edits a concept label or
adds a concept, without changing which scheme binds the column, leaves every column byte-identical.
If the enrichment identity (`content_version`, i.e. `snap.version`, cli.py's store key) is derived
from the columns alone, that shifted vocabulary would silently reuse a stale enrichment -- breaking
`content_version`'s own promise ("a shifted vocabulary invalidates downstream caches").
"""

from mnemiq.contract import CodeScheme, Column, Snapshot
from mnemiq.enrichment.pipeline import content_version
from mnemiq.ontology.records import Concept, ConceptScheme, OntologyRecords, merge_records


def _bound_column() -> Column:
    # A column bound to the MPAA scheme. The binding (id/label) is what the snapshot records;
    # the concept content lives in the OntologyRecords, never on the column.
    return Column(id="film.rating", object_id="film", name="rating",
                  code_scheme=CodeScheme(id="mpaa", label="MPAA rating"))


def _snapshot_for(onto: OntologyRecords) -> Snapshot:
    return Snapshot(version="", source_id="s", created_at="t",
                    columns=[_bound_column()], ontology_version=onto.version)


def test_certified_scheme_edit_invalidates_enrichment_identity():
    # The certified scheme changes only a concept's pref_label -- the binding id/label is untouched.
    base = OntologyRecords(schemes=[], definitions=[])
    scheme_v1 = ConceptScheme(id="mpaa", label="MPAA rating",
                              concepts=[Concept(id="r", notation="R", pref_label="Restricted")])
    scheme_v2 = ConceptScheme(id="mpaa", label="MPAA rating",
                              concepts=[Concept(id="r", notation="R", pref_label="Restricted (17+)")])
    onto_v1 = merge_records(base, [scheme_v1])
    onto_v2 = merge_records(base, [scheme_v2])
    assert onto_v1.version != onto_v2.version  # sanity: records_version already tracks concept content

    snap_v1 = _snapshot_for(onto_v1)
    snap_v2 = _snapshot_for(onto_v2)
    # Columns are byte-identical across the two runs...
    assert [c.model_dump() for c in snap_v1.columns] == [c.model_dump() for c in snap_v2.columns]
    # ...yet the enrichment identity MUST differ so the cached enrichment is invalidated.
    assert content_version(snap_v1) != content_version(snap_v2)


def test_same_ontology_yields_stable_identity():
    onto = merge_records(OntologyRecords(schemes=[], definitions=[]),
                         [ConceptScheme(id="mpaa", label="MPAA rating",
                                        concepts=[Concept(id="r", notation="R", pref_label="Restricted")])])
    assert content_version(_snapshot_for(onto)) == content_version(_snapshot_for(onto))


def test_no_ontology_leaves_identity_unchanged():
    # Non-ontology installs must not see their enrichment identity churn: an empty ontology_version
    # is excluded from the hash, so the version is identical to the pre-ontology_version behavior.
    with_field = Snapshot(version="", source_id="s", created_at="t",
                          columns=[_bound_column()], ontology_version="")
    without_field = Snapshot(version="", source_id="s", created_at="t",
                             columns=[_bound_column()])
    assert content_version(with_field) == content_version(without_field)
