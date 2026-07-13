from mnemiq.contract import CodedValue, JoinKey, MeasureExpr


def test_measure_expr_round_trip():
    m = MeasureExpr(expr="sum(arr)", source="v_revenue")
    assert MeasureExpr.model_validate_json(m.model_dump_json()) == m


def test_coded_value_and_join_key():
    assert CodedValue(code="S", meaning="suspended").meaning == "suspended"
    assert JoinKey(left="a.id", right="b.a_id").left == "a.id"
