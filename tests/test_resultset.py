from decimal import Decimal

import pyarrow as pa

from mnemiq.execute.resultset import cluster, results_equal


def _t(d):
    return pa.table(d)


def test_identical_results_are_equal():
    assert results_equal(_t({"n": [7]}), _t({"n": [7]}))


def test_a_different_number_is_not_equal():
    assert not results_equal(_t({"n": [7]}), _t({"n": [8]}))


def test_float_rounding_is_tolerated_but_integers_are_exact():
    assert results_equal(_t({"r": [0.6429]}), _t({"r": [0.6429268]}))
    assert not results_equal(_t({"n": [819]}), _t({"n": [820]}))


def test_column_order_does_not_matter():
    assert results_equal(_t({"k": ["a"], "n": [1]}), _t({"n": [1], "k": ["a"]}))


def test_row_order_does_not_matter():
    assert results_equal(_t({"n": [1, 2]}), _t({"n": [2, 1]}))


def test_different_column_count_is_not_equal():
    assert not results_equal(_t({"n": [1]}), _t({"n": [1], "x": [2]}))


def test_decimal_and_float_are_equal():
    assert results_equal(_t({"t": [Decimal("13600.00")]}), _t({"t": [13600.0]}))


def test_cluster_groups_agreeing_results_in_first_appearance_order():
    tables = [_t({"n": [7]}), _t({"n": [7]}), _t({"n": [9]}), _t({"n": [7]}), _t({"n": [9]})]
    groups = cluster(tables)
    assert groups == [[0, 1, 3], [2, 4]]


def test_cluster_of_all_distinct_is_all_singletons():
    tables = [_t({"n": [1]}), _t({"n": [2]}), _t({"n": [3]})]
    assert cluster(tables) == [[0], [1], [2]]
