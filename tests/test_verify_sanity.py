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
