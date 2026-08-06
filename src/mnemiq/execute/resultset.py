from __future__ import annotations

import math
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

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


def _cells_match(a: object, b: object, rel_tol: float) -> bool:
    if a is None or b is None:
        return a is b  # NULL is not zero, and not the empty string
    if isinstance(a, float) and isinstance(b, float):
        if a.is_integer() and b.is_integer():
            return a == b  # counts are exact; tolerance is for presentation rounding only
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)
    return a == b


# Grading tolerance, ported from beacon so the two graders answer the same question.
# Both bounds are measured, not chosen: 5e-7 is wide enough to absorb a DECIMAL-vs-REAL
# cast, and 1e-5 splits float-representation noise from genuinely different numbers with
# an order of magnitude to spare on each side. Deliberately far tighter than the 1e-2
# this replaces -- a 1% band called 1038.15 and 1039.32 the same answer.
GRADE_ABS_TOL = 5e-7
GRADE_REL_TOL = 1e-5


def grade_cells_match(a: object, b: object) -> bool:
    """The strict cell reading used for grading. Gold is `a`.

    Separate from `_cells_match` on purpose: that one is the engine's, used to cluster
    self-consistency candidates, and its tolerance is not a grading decision. Tying the
    two would mean a change to how answers are graded silently changed how they are
    produced.
    """
    if a is None or b is None:
        return a is b  # NULL is not zero, and not the empty string
    if isinstance(a, float) and isinstance(b, float):
        if a.is_integer() and b.is_integer():
            return a == b  # a count, a year or an id is right or wrong, never within 1%
        difference = abs(a - b)
        return difference <= GRADE_ABS_TOL or difference <= GRADE_REL_TOL * abs(a)
    return a == b


def is_rounding_of(value: float, other: float) -> bool:
    """Whether `value` is `other` rounded to some number of decimal places.

    A band cannot express this. "The same quantity, printed shorter" and "a different
    quantity, close by" are different claims, and a percentage accepts both -- which is
    why this is a rule of its own rather than a wider tolerance.

    Rounding to zero places is presentation only above magnitude one: 53 for 52.63 is a
    rounding, but 0.0 for 0.196 collapses a small quantity to nothing, and that answer
    did not get the fact.

    Both rounding conventions count. Python's round() is banker's rounding, so it turns
    38.125 into 38.12 and would not recognise a database that returned 38.13 -- which is
    the same quantity, rounded the other legal way. Measured on Spider local023, where
    exactly that pair was being called a wrong answer. A convention is presentation; the
    whole point of this rule is that presentation does not decide correctness.
    """
    start = 0 if abs(other) >= 1.0 else 1
    exact = Decimal(str(other))
    for places in range(start, 7):
        step = Decimal(1).scaleb(-places)
        half_even = float(exact.quantize(step, rounding=ROUND_HALF_EVEN))
        half_up = float(exact.quantize(step, rounding=ROUND_HALF_UP))
        if abs(half_even - value) <= 1e-9 or abs(half_up - value) <= 1e-9:
            return True
    return False


def facts_cells_match(a: object, b: object) -> bool:
    """The got-facts cell reading: the same value, however it was printed. Gold is `a`.

    Everything the strict reading accepts, plus a decimal rounding in either direction.
    A candidate that returns 66.62 where gold computes 66.6230 got the fact; the
    quantity is the same and only the presentation differs. The strict reading stays
    strict, because exact match is what claims two result sets are the same.
    """
    if grade_cells_match(a, b):
        return True
    if isinstance(a, float) and isinstance(b, float) and not (a.is_integer() and b.is_integer()):
        return is_rounding_of(b, a) or is_rounding_of(a, b)
    return False


def _rows(table: pa.Table) -> list[list[object]]:
    columns = table.schema.names
    return [[normalize(row[c]) for c in columns] for row in table.to_pylist()]


def _match_rows(left: list[list], right: list[list], rel_tol: float, cell_match=None) -> bool:
    """Multiset row comparison. `left` is gold when this is used for grading.

    `cell_match` takes (gold, candidate) and overrides the engine's tolerance reading;
    `rel_tol` applies only to the default one.
    """
    match = cell_match or (lambda x, y: _cells_match(x, y, rel_tol))
    if len(left) != len(right):
        return False
    remaining = list(right)
    for lrow in left:
        for i, rrow in enumerate(remaining):
            if len(lrow) == len(rrow) and all(
                match(x, y) for x, y in zip(lrow, rrow, strict=True)
            ):
                del remaining[i]  # multiset: each left row consumes one right row
                break
        else:
            return False
    return True


def results_equal(a: pa.Table, b: pa.Table, rel_tol: float = 1e-2) -> bool:
    """Do two result sets state the same facts? Strict on column count (two candidates
    agree only if they answer the same question), but blind to row order, column order,
    and float presentation noise."""
    if a.num_columns != b.num_columns:
        return False
    ra, rb = _rows(a), _rows(b)
    if _match_rows(ra, rb, rel_tol):
        return True
    # column order is not meaning: canonicalize each row's cells and retry
    sa = [sorted(r, key=repr) for r in ra]
    sb = [sorted(r, key=repr) for r in rb]
    return _match_rows(sa, sb, rel_tol)


def cluster(tables: list[pa.Table]) -> list[list[int]]:
    """Group result sets that agree. Greedy; groups in first-appearance order."""
    groups: list[list[int]] = []
    for i, table in enumerate(tables):
        for group in groups:
            if results_equal(tables[group[0]], table):
                group.append(i)
                break
        else:
            groups.append([i])
    return groups
