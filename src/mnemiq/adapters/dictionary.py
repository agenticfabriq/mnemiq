"""What a source's data dictionary says about a table, read without reading the table's data.

Oracle keeps these as its optimizer statistics (`ALL_TAB_STATISTICS`), gathered by its own maintenance job or by
`DBMS_STATS`. They are estimates as of when they were gathered, and a reader must treat them so.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableSize:
    """A table's size as the dictionary last recorded it. None where the dictionary holds nothing: a table never
    analysed has no statistics, and nothing here may be read as zero."""

    rows: int | None
    avg_row_len: int | None  # bytes
    analyzed: str | None  # the date its statistics were gathered, ISO
    # True when the dictionary says the statistics are stale, False when it says they are not, None when it does not
    # say: unknown is never "not stale".
    stale: bool | None
