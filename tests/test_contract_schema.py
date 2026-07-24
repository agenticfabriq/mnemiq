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
