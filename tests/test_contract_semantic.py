from mnemiq.contract import (
    Column,
    MeasureExpr,
    Metric,
    Relationship,
    Snapshot,
)


def test_metric_with_measure_and_dimensions():
    m = Metric(
        id="revenue",
        label="Revenue",
        status="certified",
        owner="finance",
        grain="policy",
        measure=MeasureExpr(expr="sum(premium_amount)", source="v_premium"),
        time_dimension="effective_date",
        compatible_dimensions=["product", "region"],
    )
    assert Metric.model_validate_json(m.model_dump_json()) == m


def test_column_defaults_empty_coded_values():
    c = Column(id="c1", object_id="party", name="status")
    assert c.coded_values == []
    assert c.pii_level is None


def test_relationship_from_alias_round_trips():
    r = Relationship.model_validate(
        {"id": "r1", "from": "claim", "to": "party", "cardinality": "many_to_one"}
    )
    assert r.from_ == "claim"
    dumped = r.model_dump(by_alias=True)
    assert dumped["from"] == "claim"


def test_snapshot_aggregates_and_defaults():
    s = Snapshot(version="v1", source_id="acme", created_at="2026-07-12T00:00:00Z")
    assert s.metrics == [] and s.columns == [] and s.relationships == []
