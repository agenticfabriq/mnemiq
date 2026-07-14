from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal

import pyarrow as pa

_NUMBER = (int, float, Decimal)


def normalize(value: object) -> object:
    """Reduce a cell to what it *means*, discarding how it was spelled."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value  # before the int check: a bool is an int in Python
    if isinstance(value, _NUMBER):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value).strip()


def _cells_match(gold: object, candidate: object, rel_tol: float) -> bool:
    if gold is None or candidate is None:
        return gold is candidate  # NULL is not zero, and not the empty string
    if isinstance(gold, float) and isinstance(candidate, float):
        if gold.is_integer() and candidate.is_integer():
            # A count of 819 is not "within 1%" of 820 -- it is wrong. Tolerance exists
            # for presentation rounding, and integers are never presentation-rounded.
            return gold == candidate
        return math.isclose(gold, candidate, rel_tol=rel_tol, abs_tol=1e-9)
    return gold == candidate


def _rows(table: pa.Table) -> list[list[object]]:
    columns = table.schema.names
    return [[normalize(row[c]) for c in columns] for row in table.to_pylist()]


def _match_rows(gold: list[list], candidate: list[list], rel_tol: float) -> bool:
    if len(gold) != len(candidate):
        return False

    remaining = list(candidate)
    for gold_row in gold:
        for i, candidate_row in enumerate(remaining):
            if len(gold_row) == len(candidate_row) and all(
                _cells_match(g, c, rel_tol) for g, c in zip(gold_row, candidate_row, strict=True)
            ):
                del remaining[i]  # a multiset match: each gold row consumes one candidate row
                break
        else:
            return False
    return True


def results_match(gold: pa.Table, candidate: pa.Table, rel_tol: float = 1e-2) -> bool:
    """Do these two result sets state the same facts?

    Not: are they the same query, the same column names, the same row order, or the same
    formatting. A different query that returns the right answer is correct -- that is the
    whole point of result-based grading (spec 6.9). What must match is the data.
    """
    if gold.num_columns != candidate.num_columns:
        return False

    gold_rows, candidate_rows = _rows(gold), _rows(candidate)
    if _match_rows(gold_rows, candidate_rows, rel_tol):
        return True

    # Column order is not meaning: `SELECT k, count(*)` and `SELECT count(*), k` are the same
    # answer. Retry with each row's values sorted into a canonical order.
    def sort_cells(rows: list[list[object]]) -> list[list[object]]:
        return [sorted(row, key=repr) for row in rows]

    return _match_rows(sort_cells(gold_rows), sort_cells(candidate_rows), rel_tol)
