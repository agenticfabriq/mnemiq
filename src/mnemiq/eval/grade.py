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
    for keep in column_choices:
        projected = candidate.select(list(keep))
        candidate_rows = _rows(projected)
        if _match_rows(gold_rows, candidate_rows, rel_tol):
            return True

        # Column order is not meaning: `SELECT k, count(*)` and `SELECT count(*), k` are
        # the same answer. Retry with each row's values sorted into a canonical order.
        def sort_cells(rows: list[list[object]]) -> list[list[object]]:
            return [sorted(row, key=repr) for row in rows]

        if _match_rows(sort_cells(gold_rows), sort_cells(candidate_rows), rel_tol):
            return True
    return False
