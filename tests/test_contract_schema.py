import json

from mnemiq.contract import export_json_schema


def test_schema_has_entry_points_and_metric_fields():
    schema = export_json_schema()
    assert set(schema) >= {"snapshot", "trace", "identity_context"}
    defs = schema["snapshot"].get("$defs", {})
    assert "Metric" in defs
    assert "measure" in defs["Metric"]["properties"]


def test_schema_contains_no_paid_concepts():
    # the open contract must never carry governance/grading fields
    blob = json.dumps(export_json_schema()).lower()
    for banned in (
        "policy",
        "policies",
        "proposal",
        "review_decision",
        "certification",
        "grading",
        "drift",
    ):
        assert banned not in blob, f"paid concept leaked into open contract: {banned}"


def test_committed_schema_is_up_to_date(tmp_path):
    from mnemiq.contract.schema import write_schema

    out = tmp_path / "s.json"
    write_schema(str(out))
    committed = json.load(open("schemas/mnemiq-contract.schema.json"))
    assert json.load(open(out)) == committed  # regenerate + commit if this fails


def test_schema_is_semver_stamped():
    import re

    from mnemiq.contract import export_json_schema

    version = export_json_schema()["format_version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version


def test_schema_describes_the_whole_pinned_format():
    from mnemiq.contract import export_json_schema

    schema = export_json_schema()
    assert {"snapshot", "trace", "identity_context",
            "ontology_records", "certified_record"} <= set(schema)

    # ontology types are now exported (the SP1 gap), and the record wrapper is present
    onto_defs = set(schema["ontology_records"].get("$defs", {}))
    assert {"Concept", "ConceptScheme"} <= onto_defs

    rec_defs = set(schema["certified_record"].get("$defs", {}))
    assert {"RecordEnvelope", "Provenance", "Column", "ConceptScheme"} <= rec_defs


def test_trust_marker_is_present_but_no_governance_leaks():
    import json

    from mnemiq.contract import export_json_schema

    blob = json.dumps(export_json_schema()).lower()
    assert "certifier" in blob and "evidence_ref" in blob  # the trust marker made it in
    # the governance term and the rest of the paid layer stay out
    for banned in ("certification", "policy", "proposal", "review_decision", "grading", "drift"):
        assert banned not in blob, f"paid concept leaked: {banned}"


# --- an estimate apart from the counts: `Column.estimate`, `Column.declared_unique` ----------------

def _snapshot(**orders_id):
    """A snapshot as enrich writes one today: counts, codes, a partition fact, a join, the jobs the version reads."""
    from mnemiq.contract.semantic import Column, Job, Relationship, Snapshot, SourceBinding
    from mnemiq.contract.values import CodedValue, JoinKey

    return Snapshot(version="", source_id="shop", created_at="2026-11-20T00:00:00+00:00", columns=[
        Column(id="orders.id", object_id="orders", name="id", data_type="NUMBER", row_count=1000,
               distinct_count=1000, null_count=0, **orders_id),
        Column(id="orders.status", object_id="orders", name="status", data_type="VARCHAR2", row_count=1000,
               distinct_count=2, null_count=0, coded_values=[CodedValue(code="A"), CodedValue(code="B")]),
        Column(id="customers.is_current", object_id="customers", name="is_current", data_type="NUMBER",
               row_count=50, distinct_count=2, null_count=0, unique_within={"=1": ["customer_id"]}),
    ], relationships=[Relationship(id="r1", from_="orders", to="customers", cardinality="many_to_one",
                                   join_keys=[JoinKey(left="customer_id", right="customer_id")])],
       source_bindings=[SourceBinding(id="sb:orders", source_id="shop", object_id="orders", source_object="orders",
                                      binding_type="table")],
       jobs=[Job(id="profile:partitions", source_id="shop", kind="profile:partitions", status="done")])


# content_version of `_snapshot()` on main 5f0ca71a, before `estimate` and `declared_unique` existed.
VERSION_BEFORE_ESTIMATES = "4bdc976a4faf"


def test_a_snapshot_without_estimates_keeps_its_version():
    """Every snapshot written before the two fields keeps its version, so no cache or replica sees a change."""
    from mnemiq.enrichment.pipeline import content_version

    assert content_version(_snapshot()) == VERSION_BEFORE_ESTIMATES


def test_an_estimate_moves_the_version():
    from mnemiq.contract.semantic import ColumnEstimate
    from mnemiq.enrichment.pipeline import content_version

    estimated = _snapshot(estimate=ColumnEstimate(rows=1000, distinct=990, nulls=0, analyzed="2026-11-01",
                                                  sampled=False, stale=False))
    assert content_version(estimated) != content_version(_snapshot())
    later = _snapshot(estimate=ColumnEstimate(rows=1000, distinct=990, nulls=0, analyzed="2026-11-02",
                                              sampled=False, stale=False))
    assert content_version(later) != content_version(estimated)


def test_declared_unique_moves_the_version():
    from mnemiq.enrichment.pipeline import content_version

    versions = {content_version(_snapshot(**kw)) for kw in ({}, {"declared_unique": True},
                                                            {"declared_unique": False})}
    assert len(versions) == 3


def test_an_estimate_round_trips_beside_empty_exact_counts():
    """Ruling 1: an estimate survives a save and a load, and leaves the exact counts None, which every reader of them,
    and an engine predating the field, reads as "not profiled"."""
    from mnemiq.contract.semantic import Column, ColumnEstimate

    col = Column(id="orders.id", object_id="orders", name="id",
                 estimate=ColumnEstimate(rows=1000, distinct=990, nulls=0, analyzed="2026-11-01", sampled=False,
                                         stale=False), declared_unique=True)
    assert Column.model_validate_json(col.model_dump_json()) == col
    assert (col.row_count, col.distinct_count, col.null_count) == (None, None, None)
