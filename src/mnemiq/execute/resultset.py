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


def _cells_match(a: object, b: object, rel_tol: float) -> bool:
    if a is None or b is None:
        return a is b  # NULL is not zero, and not the empty string
    if isinstance(a, float) and isinstance(b, float):
        if a.is_integer() and b.is_integer():
            return a == b  # counts are exact; tolerance is for presentation rounding only
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)
    return a == b


def _rows(table: pa.Table) -> list[list[object]]:
    columns = table.schema.names
    return [[normalize(row[c]) for c in columns] for row in table.to_pylist()]


def _match_rows(left: list[list], right: list[list], rel_tol: float) -> bool:
    if len(left) != len(right):
        return False
    remaining = list(right)
    for lrow in left:
        for i, rrow in enumerate(remaining):
            if len(lrow) == len(rrow) and all(
                _cells_match(x, y, rel_tol) for x, y in zip(lrow, rrow, strict=True)
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
