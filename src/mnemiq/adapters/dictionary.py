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


@dataclass(frozen=True)
class ColumnStatistics:
    """A column's optimizer statistics (`ALL_TAB_COL_STATISTICS`). None where the dictionary holds nothing."""

    column: str
    distinct: int | None
    nulls: int | None
    histogram: str  # NONE, FREQUENCY, TOP-FREQUENCY, HEIGHT BALANCED or HYBRID
    buckets: int | None
    sample_size: int | None  # the rows the statistics were read from
    analyzed: str | None  # ISO date
    # The lowest and highest values, as the source stores them (raw bytes; Oracle cuts them at 64 bytes).
    low: bytes | None
    high: bytes | None
    # The column's declared scale (`ALL_TAB_COLUMNS.DATA_SCALE`): a NUMBER with a scale above 0 comes from the driver as
    # a float, which decides how its values are written.
    scale: int | None = None


@dataclass(frozen=True)
class HistogramEndpoint:
    """One endpoint of a column's histogram (`ALL_TAB_HISTOGRAMS`), in endpoint order.

    `number` is the running count of rows up to this endpoint, so on a frequency histogram each value's frequency is
    the difference from the previous one. `value` is Oracle's numeric form of the value: the number itself on a numeric
    column, and on a character column its leading bytes as an integer rounded to 15 significant digits. `actual` and
    `actual_raw` hold the value's text where Oracle stores it, which on a character column it often does not."""

    number: int
    value: int | float | None
    actual: str | None
    actual_raw: bytes | None
