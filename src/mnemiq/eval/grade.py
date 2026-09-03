from __future__ import annotations

from itertools import combinations

import pyarrow as pa

from mnemiq.execute.resultset import _match_rows, _rows, normalize

# normalize is re-exported: harness.py imports it from here. _rows/_match_rows back
# results_match below; the shared definitions live in the engine's resultset module.
__all__ = ["normalize", "results_match"]


def results_match(
    gold: pa.Table,
    candidate: pa.Table,
    rel_tol: float = 1e-2,
    allow_extra_columns: bool = True,
) -> bool:
    """Do these two result sets state the same facts?

    Not: are they the same query, the same column names, the same row order, or the same
    formatting. A different query that returns the right answer is correct -- that is the
    whole point of result-based grading (spec 6.9). What must match is the data.

    With allow_extra_columns=False the candidate must have exactly the gold's columns
    (BIRD execution accuracy); by default it may carry extra context columns (ACME's
    conversational questions -- the date next to the policy number it was asked for).
    """
    if allow_extra_columns:
        # One-directional tolerance: the candidate may ADD context columns but never omit a
        # gold column. Extra columns cannot rescue wrong rows -- every gold row still matches.
        if gold.num_columns > candidate.num_columns:
            return False
        column_choices = combinations(range(candidate.num_columns), gold.num_columns)
    else:
        if gold.num_columns != candidate.num_columns:
            return False
        column_choices = [tuple(range(candidate.num_columns))]

    gold_rows = _rows(gold)

    # Normalize the candidate ONCE and project the normalized cells, rather than calling
    # _rows() -- and so normalize() on every cell -- again for each column choice. The
    # choices are combinatorial (C(candidate_cols, gold_cols)), so re-normalizing inside
    # the loop multiplies the same work by the number of projections: a 20-column SELECT *
    # graded against a 5-column gold over 1000 rows is 15,504 projections and ~77M
    # normalize calls, which is hours, not milliseconds. Found when a BEAVER nova case
    # burned 426 CPU-minutes inside normalize without finishing.
    all_candidate_rows = _rows(candidate)
    # repr() is the sort key for the column-order retry below, and it was being recomputed
    # for every cell of every projection -- the same combinatorial multiplier that made
    # normalize() the first bottleneck. Computed once per cell here; the projection then
    # sorts precomputed keys.
    all_candidate_keys = [[repr(cell) for cell in row] for row in all_candidate_rows]

    # Column order is not meaning: `SELECT k, count(*)` and `SELECT count(*), k` are
    # the same answer. Retry with each row's values sorted into a canonical order.
    def sort_cells(rows: list[list[object]]) -> list[list[object]]:
        return [sorted(row, key=repr) for row in rows]

    sorted_gold_rows = sort_cells(gold_rows)

    for keep in column_choices:
        candidate_rows = [[row[i] for i in keep] for row in all_candidate_rows]
        if _match_rows(gold_rows, candidate_rows, rel_tol):
            return True
        sorted_candidate_rows = [
            [value for _, value in sorted(
                ((keys[i], row[i]) for i in keep), key=lambda pair: pair[0])]
            for row, keys in zip(all_candidate_rows, all_candidate_keys, strict=True)
        ]
        if _match_rows(sorted_gold_rows, sorted_candidate_rows, rel_tol):
            return True
    return False
