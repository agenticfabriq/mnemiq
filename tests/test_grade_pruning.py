"""The got-facts search is pruned, not changed (register M113).

`results_match` used to try every choice of the candidate's columns -- C(candidate, gold) of
them, 4e12 for a 74-column answer to an 11-column gold -- and hung. It now skips choices that
cannot match. These tests hold it to the old verdict: `_brute_force` below is the search as it
was, and the two must agree on every input, including the awkward ones the pruning has to get
right (tolerance edges, roundings in both directions, NULLs, bools, duplicates, NaN).
"""
from __future__ import annotations

import random
import time
from itertools import combinations

import pyarrow as pa
import pytest

from mnemiq.eval.grade import _distinct_rows, results_match
from mnemiq.execute.resultset import _match_rows, _rows, facts_cells_match, grade_cells_match


def _brute_force(gold: pa.Table, candidate: pa.Table, allow_extra_columns: bool = True,
                 duplicate_rows_insignificant: bool = False) -> bool:
    """The search before pruning, kept verbatim as the oracle."""
    cell_match = facts_cells_match if allow_extra_columns else grade_cells_match
    if allow_extra_columns:
        if gold.num_columns > candidate.num_columns:
            return False
        column_choices = combinations(range(candidate.num_columns), gold.num_columns)
    else:
        if gold.num_columns != candidate.num_columns:
            return False
        column_choices = [tuple(range(candidate.num_columns))]
    gold_rows = _rows(gold)
    if duplicate_rows_insignificant:
        gold_rows = _distinct_rows(gold_rows)
    all_rows = _rows(candidate)
    all_keys = [[repr(cell) for cell in row] for row in all_rows]
    sorted_gold: list[list[object]] = []
    if allow_extra_columns:
        sorted_gold = [sorted(row, key=repr) for row in gold_rows]
        if duplicate_rows_insignificant:
            sorted_gold = _distinct_rows(sorted_gold)
    for keep in column_choices:
        candidate_rows = [[row[i] for i in keep] for row in all_rows]
        if duplicate_rows_insignificant:
            candidate_rows = _distinct_rows(candidate_rows)
        if _match_rows(gold_rows, candidate_rows, 0.0, cell_match):
            return True
        if not allow_extra_columns:
            continue
        sorted_candidate = [
            [value for _, value in sorted(((keys[i], row[i]) for i in keep), key=lambda p: p[0])]
            for row, keys in zip(all_rows, all_keys, strict=True)
        ]
        if duplicate_rows_insignificant:
            sorted_candidate = _distinct_rows(sorted_candidate)
        if _match_rows(sorted_gold, sorted_candidate, 0.0, cell_match):
            return True
    return False


# Values chosen to sit on the edges the pruning must not cross: a rounding either way, a value
# just inside and just outside the numeric tolerance, integers that must stay exact, a bool
# that Python calls equal to 1.0, NULL, NaN, and strings.
_POOL = [0.0, 1.0, 2.0, 1.4249, 1.42, 1.425, 1.43, 1.4, 1.0000004, 1.000006, 52.63, 53.0, 0.196,
         0.2, 0.0, 38.125, 38.12, 38.13, 1e6, 1e6 + 5.0, 1e6 + 20.0, 7.0, None, True, False, "a",
         "b", "1", float("nan"), -1.5, -2.0,
         # Within the 1e-5 relative tolerance of each other but no rounding of each other --
         # the only pairs that test the tolerance window on its own -- and one just outside.
         1234567.5, 1234570.25, 1234599.5]


def _col(rng: random.Random, n: int) -> list[object]:
    kind = rng.random()
    if kind < 0.15:  # a constant column: many choices look alike
        v = rng.choice(_POOL)
        return [v] * n
    return [rng.choice(_POOL) for _ in range(n)]


def _table(cols: list[list[object]]) -> pa.Table:
    # Mixed types do not fit one arrow column, so carry them as python objects via strings
    # only where needed: arrow infers a type per column, and a mixed column falls back here.
    arrays = []
    for col in cols:
        try:
            arrays.append(pa.array(col))
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            arrays.append(pa.array([None if v is None else str(v) for v in col]))
    return pa.Table.from_arrays(arrays, names=[f"c{i}" for i in range(len(cols))])


def _case(rng: random.Random, extra_columns: bool) -> tuple[pa.Table, pa.Table]:
    rows = rng.choice([0, 1, 1, 2, 3, 4])
    gold_cols = [_col(rng, rows) for _ in range(rng.choice([0, 1, 1, 2, 2, 3]))]
    gold = _table(gold_cols) if gold_cols else pa.table({})
    # Exact match needs the same width, so there the candidate is mostly gold's own columns.
    width = rng.choice([1, 2, 3, 4, 5, 6]) if extra_columns or rng.random() < 0.3 else 0
    cand_cols = [_col(rng, rows) for _ in range(width)]
    # Often plant gold's columns in the candidate -- reordered, rounded, or with rows shuffled --
    # so that matches are common enough to test the True side too.
    if gold_cols and rng.random() < 0.7:
        planted = list(gold_cols)
        if rng.random() < 0.3:
            planted = [[round(v, 2) if isinstance(v, float) and v == v else v for v in c]
                       for c in planted]
        slots = sorted(rng.sample(range(width + len(planted)), len(planted)))
        if rng.random() < 0.3:
            rng.shuffle(slots)
        for slot, col in zip(slots, planted, strict=True):
            cand_cols.insert(min(slot, len(cand_cols)), col)
        if rows > 1 and rng.random() < 0.4:  # the same rows in a different order
            order = list(range(rows))
            rng.shuffle(order)
            cand_cols = [[c[i] for i in order] for c in cand_cols]
    return gold, _table(cand_cols)


@pytest.mark.parametrize("allow_extra", [True, False])
@pytest.mark.parametrize("dupes", [False, True])
def test_the_pruned_search_agrees_with_the_full_one(allow_extra, dupes):
    rng = random.Random(1113 + 2 * allow_extra + dupes)
    agree = matched = 0
    for _ in range(4000):
        gold, cand = _case(rng, allow_extra)
        expected = _brute_force(gold, cand, allow_extra, dupes)
        got = results_match(gold, cand, allow_extra, dupes)
        assert got == expected, (gold.to_pydict(), cand.to_pydict(), allow_extra, dupes)
        agree += 1
        matched += expected
    # A sampler that never produced a match would test only the False side.
    assert matched > agree * 0.1, f"only {matched} of {agree} cases matched"


def test_a_wide_answer_holding_every_gold_column_is_graded_and_fast():
    """The shape that hung: one row, gold's 11 columns among 74. C(74, 11) is 4e12 choices."""
    rng = random.Random(7)
    gold_vals = {f"g{i}": [round(rng.uniform(0, 100), 4)] for i in range(11)}
    cand = {f"x{i}": [round(rng.uniform(100, 200), 4)] for i in range(63)}
    cand.update({f"c_{k}": v for k, v in gold_vals.items()})
    start = time.perf_counter()
    assert results_match(pa.table(gold_vals), pa.table(cand), allow_extra_columns=True)
    assert time.perf_counter() - start < 2.0


def test_a_wide_wrong_answer_is_refused_fast():
    """And the False side, where the old search had to try every choice before giving up."""
    gold = pa.table({f"g{i}": [float(i) + 0.5] for i in range(11)})
    cand = pa.table({f"x{i}": [float(i) + 0.25] for i in range(74)})
    start = time.perf_counter()
    assert not results_match(gold, cand, allow_extra_columns=True)
    assert time.perf_counter() - start < 2.0


def test_a_wide_answer_under_the_cell_sorted_reading_is_graded():
    """Gold's columns present but in a different order: only the cell-sorted retry finds it."""
    gold = pa.table({"k": ["a", "b"], "n": [3.0, 5.0]})
    cand = {f"x{i}": [f"z{i}", f"y{i}"] for i in range(40)}
    cand.update({"count": [3.0, 5.0], "key": ["a", "b"]})
    assert results_match(gold, pa.table(cand), allow_extra_columns=True)
