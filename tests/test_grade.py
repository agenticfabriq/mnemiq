from datetime import datetime
from decimal import Decimal

import pyarrow as pa

from mnemiq.eval.grade import results_match


def _t(data: dict) -> pa.Table:
    return pa.table(data)


def test_an_identical_result_matches():
    assert results_match(_t({"n": [820]}), _t({"n": [820]}))


def test_a_different_column_name_still_matches():
    # the model aliases freely; grading names would be string matching by the back door
    assert results_match(_t({"n": [820]}), _t({"total_claims": [820]}))


def test_a_different_row_order_still_matches():
    gold = _t({"k": ["no", "yes"], "n": [128, 692]})
    candidate = _t({"k": ["yes", "no"], "n": [692, 128]})
    assert results_match(gold, candidate)


def test_a_different_column_order_still_matches():
    gold = _t({"k": ["yes"], "n": [692]})
    candidate = _t({"n": [692], "k": ["yes"]})
    assert results_match(gold, candidate)


def test_a_wrong_number_fails():
    assert not results_match(_t({"n": [820]}), _t({"n": [819]}))


def test_a_missing_row_fails():
    gold = _t({"k": ["no", "yes"], "n": [128, 692]})
    candidate = _t({"k": ["yes"], "n": [692]})
    assert not results_match(gold, candidate)


def test_presentation_rounding_is_tolerated():
    # 0.6429 -> 0.64 is a formatting choice, not a computational error
    assert results_match(_t({"r": [0.6429268]}), _t({"r": [0.6429]}))


def test_a_materially_different_number_is_not_tolerated():
    # a wrong filter or a missing DISTINCT moves a number far more than 1%
    assert not results_match(_t({"r": [0.6429]}), _t({"r": [0.58]}))


def test_decimal_and_float_are_the_same_number():
    assert results_match(_t({"total": [Decimal("13600.00")]}), _t({"total": [13600.0]}))


def test_an_integer_and_its_float_are_the_same_number():
    assert results_match(_t({"n": [820]}), _t({"n": [820.0]}))


def test_dates_compare_by_value():
    gold = _t({"d": [datetime(2019, 1, 15)]})
    assert results_match(gold, _t({"d": [datetime(2019, 1, 15)]}))
    assert not results_match(gold, _t({"d": [datetime(2019, 1, 16)]}))


def test_an_empty_result_matches_an_empty_result():
    empty = pa.table({"n": pa.array([], type=pa.int64())})
    assert results_match(empty, empty)


def test_an_empty_result_does_not_match_a_populated_one():
    empty = pa.table({"n": pa.array([], type=pa.int64())})
    assert not results_match(empty, _t({"n": [1]}))


def test_a_null_matches_a_null():
    nulls = pa.table({"n": pa.array([None], type=pa.int64())})
    assert results_match(nulls, nulls)
    assert not results_match(nulls, _t({"n": [0]}))  # NULL is not zero


def test_an_extra_context_column_does_not_make_a_right_answer_wrong():
    # "which policy is earliest?" answered with the policy number AND its date is not an
    # error -- the fact asked for is there. Found live: three of four eval "failures" were
    # right answers carrying one extra column.
    gold = _t({"policy_number": ["31003000336"]})
    candidate = _t({"policy_number": ["31003000336"], "effective_date": ["2015-01-01"]})
    assert results_match(gold, candidate)


def test_extra_columns_cannot_rescue_wrong_rows():
    gold = _t({"n": [820]})
    candidate = _t({"n": [819], "extra": ["context"]})
    assert not results_match(gold, candidate)


def test_extra_columns_cannot_rescue_a_missing_row():
    gold = _t({"d": ["2019-01-15", "2019-06-02"]})
    candidate = _t({"id": [1], "d": ["2019-01-15"]})
    assert not results_match(gold, candidate)


def test_the_gold_may_never_have_more_columns_than_the_candidate():
    # the tolerance is one-directional: the candidate may add context, never omit facts
    gold = _t({"k": ["yes"], "n": [692]})
    candidate = _t({"n": [692]})
    assert not results_match(gold, candidate)
