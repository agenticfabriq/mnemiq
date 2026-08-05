import pyarrow as pa

from mnemiq.execute.render import render_result


def _table(rows: int, value: str = "x") -> pa.Table:
    return pa.table({"n": list(range(rows)), "label": [value] * rows})


def test_a_small_result_renders_every_row():
    text = render_result(_table(3))
    assert "n" in text and "label" in text
    assert text.count("\n") >= 3
    assert "truncated" not in text.lower()


def test_a_large_result_says_it_is_truncated():
    text = render_result(_table(500), max_rows=10)
    assert "500" in text  # the true row count is stated
    assert "10" in text  # and how many are shown
    assert "truncat" in text.lower()  # never let the model summarize a partial view silently


def test_a_wide_cell_is_clipped():
    text = render_result(_table(1, value="y" * 500), max_cell=20)
    assert "y" * 21 not in text


def test_an_empty_result_is_stated_plainly():
    text = render_result(pa.table({"n": pa.array([], type=pa.int64())}))
    assert "0 rows" in text  # "no rows" is a real answer, and must not read as an error


def test_nulls_render_as_null_not_none():
    table = pa.table({"n": pa.array([None, 1], type=pa.int64())})
    assert "NULL" in render_result(table)


def test_a_shortened_cell_is_declared_not_just_marked():
    # The model read a bare `…` as evidence and reported rows truncated when only cells
    # were. Row truncation was always announced; cell truncation was not.
    t = pa.table({"detail": ["x" * 400]})
    out = render_result(t)

    assert "…" in out
    assert "shortened for this prompt" in out
    assert "(1 rows)" in out, "the row count must still be honest"


def test_nothing_is_declared_when_nothing_was_shortened():
    out = render_result(pa.table({"n": [1, 2]}))
    assert "shortened" not in out
    assert "…" not in out


def test_ordinary_values_are_no_longer_cut():
    # 120 halved schema/description columns; a policy number or a sentence must survive.
    value = "a claim description of the kind an adjuster writes, " * 3
    assert len(value) < 240
    assert value in render_result(pa.table({"note": [value]}))


def test_both_bounds_are_declared_together():
    rows = ["y" * 400] * 60
    out = render_result(pa.table({"detail": rows}), max_rows=50)
    assert "showing 50 of 60 rows (truncated)" in out
    assert "shortened for this prompt" in out
