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


def test_evaluation_case_carries_gold_sql_not_a_gold_string():
    from mnemiq.contract import EvaluationCase

    case = EvaluationCase(
        id="fire-count",
        question="how many fire claims are there?",
        gold_sql="SELECT count(*) AS n FROM fireclaim",
        tags=["aggregate"],
    )
    assert case.answerable is True  # the default
    assert case.expected_answer is None  # documentation only; nothing grades on it
    assert EvaluationCase.model_validate_json(case.model_dump_json()) == case


def test_a_column_carries_its_profile_counts():
    # Profiling computes these; the eval proved that dropping them at the contract
    # boundary turns an empty column into an invisible trap (COUNT(DISTINCT claimnumber)).
    c = Column(
        id="fireclaim.claimnumber",
        object_id="fireclaim",
        name="claimnumber",
        row_count=820,
        distinct_count=0,
        null_count=820,
    )
    assert c.null_count == c.row_count == 820
    assert Column.model_validate_json(c.model_dump_json()) == c


def test_the_counts_default_to_absent_not_zero():
    c = Column(id="t.c", object_id="t", name="c")
    assert c.row_count is None and c.distinct_count is None and c.null_count is None


def test_an_unanswerable_case_has_no_gold_sql():
    from mnemiq.contract import EvaluationCase

    case = EvaluationCase(
        id="no-salary",
        question="what is the average salary of our employees?",
        answerable=False,
    )
    assert case.gold_sql is None  # there is no right query, only a right refusal
