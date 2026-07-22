from mnemiq.contract import CodedValue, JoinKey, MeasureExpr


def test_measure_expr_round_trip():
    m = MeasureExpr(expr="sum(arr)", source="v_revenue")
    assert MeasureExpr.model_validate_json(m.model_dump_json()) == m


def test_coded_value_and_join_key():
    assert CodedValue(code="S", meaning="suspended").meaning == "suspended"
    assert JoinKey(left="a.id", right="b.a_id").left == "a.id"


def test_coded_value_source_defaults_none_and_accepts_provenance():
    assert CodedValue(code="A").source is None
    assert CodedValue(code="A", meaning="detect", source="lookup").source == "lookup"
