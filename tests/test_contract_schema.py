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
