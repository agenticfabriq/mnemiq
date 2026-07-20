from __future__ import annotations

import math

import pyarrow as pa

from mnemiq.verify.verdict import VerifyVerdict


def _degenerate(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def sanity_check(question: str, table: pa.Table) -> VerifyVerdict | None:
    """Defer results that cannot be a real answer: no rows, a null/NaN scalar, or all-null cells.
    An empty result presented as an answer is the single biggest catchable wrong-class (~17%)."""
    if table.num_rows == 0:
        return VerifyVerdict(0.0, True, "The query returned no rows, so I can't answer this confidently.", "sanity")
    rows = table.to_pylist()
    if table.num_rows == 1 and table.num_columns == 1:
        if _degenerate(next(iter(rows[0].values()))):
            return VerifyVerdict(0.0, True, "The query produced an empty value, so I can't answer this confidently.", "sanity")
    if all(all(v is None for v in r.values()) for r in rows):
        return VerifyVerdict(0.0, True, "The query returned only empty values, so I can't answer this confidently.", "sanity")
    return None
