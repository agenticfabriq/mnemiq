import pyarrow as pa

from mnemiq.verify.sanity import sanity_check


def test_empty_result_defers():
    v = sanity_check("how many?", pa.table({"n": pa.array([], type=pa.int64())}))
    assert v is not None and v.defer and v.layer == "sanity" and v.confidence == 0.0


def test_single_null_scalar_defers():
    v = sanity_check("avg?", pa.table({"avg": [None]}))
    assert v is not None and v.defer


def test_single_nan_scalar_defers():
    v = sanity_check("avg?", pa.table({"avg": [float("nan")]}))
    assert v is not None and v.defer


def test_all_null_rows_defer():
    v = sanity_check("list", pa.table({"a": [None, None], "b": [None, None]}))
    assert v is not None and v.defer


def test_normal_result_passes():
    assert sanity_check("q", pa.table({"name": ["Alice"], "n": [3]})) is None


def test_a_value_found_in_another_column_is_its_own_clause():
    """`repair_hints`: each missing value says where it does appear, so two found in different places are two clauses."""
    from mnemiq.sql.values_check import ABSENT, FilterValue, ValueGrounding

    grounding = ValueGrounding(values=(
        FilterValue(text="B-17", outcome=ABSENT, why="probed", column="orders.customer_ref",
                    placed=("orders", "customer_ref"), found_in=("orders.batch_ref",)),
        FilterValue(text="C-9", outcome=ABSENT, why="probed", column="orders.customer_ref",
                    placed=("orders", "customer_ref"), found_in=("shipments.ref", "returns.ref"))),
        never_true=None, filters=())
    verdict = sanity_check("q", pa.table({"id": pa.array([], type=pa.int64())}), grounding)
    assert verdict.reason == (
        "No rows matched, and 'B-17' does not appear in orders.customer_ref, though it does appear in orders.batch_ref; "
        "'C-9' does not appear in orders.customer_ref, though it does appear in shipments.ref and returns.ref, so I "
        "can't tell a true \"none\" from a misspelled value.")
