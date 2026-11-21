"""A table's profile from the source's optimizer statistics, reading none of its rows.

The exact profile (`profiling.profile_table`) reads every table in full several times, which on a production database's
large tables runs for hours. This reads what the database already keeps about its tables: row and null counts, an
approximate distinct count, the keys it declares and enforces, and the histograms its optimizer built.

Three rules keep an estimate from passing for a fact:
- the counts go into `ColumnEstimate`, never into the exact counts;
- a column is unique only where a key declares it (`unique_columns`), never because its distinct count is near its rows;
- a code list is given only where a frequency histogram covers every row and every value reads back exactly
  (`frequency_codes`), and is otherwise empty, never partial. Readers treat a column's codes as its whole vocabulary,
  and the definitions' meanings are kept only for codes in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

import logging

from mnemiq.adapters.dictionary import ColumnStatistics, HistogramEndpoint
from mnemiq.catalog import TableInfo, is_key_like, is_sensitive_name
from mnemiq.contract.semantic import ColumnEstimate

# Oracle keeps at most 64 bytes of a value in a histogram's text and in LOW_VALUE / HIGH_VALUE (measured on 23.26).
_TEXT_BYTES_KEPT = 64
# A cut inside a character of up to 4 bytes leaves at least 61 bytes of it.
_NON_ASCII_TEXT_MAY_BE_CUT_AT = _TEXT_BYTES_KEPT - 3

# On a character column without text, ENDPOINT_VALUE is the value's first 15 bytes as a big-endian integer, rounded to
# 15 significant digits (measured on 23.26). For a printable ASCII first byte the integer has 36 digits, so a rounding
# step is 10**21. Two texts of at most 5 bytes differ by at least 256**10 and never round alike, and a longer text with
# the same first 5 bytes adds at least 256**9 (~4.7e21, several steps) unless its sixth byte is NUL, so it cannot round
# onto the shorter one. Two kinds of value can:
# - a sixth byte of 0xFF, with large bytes after it, sits just below the next 5-byte value and rounds onto it ('ABCDD'
#   + 0xFF 0xEF 0xFF... onto 'ABCDE'). 0xFF is a character in a single-byte set ('ÿ' in WE8MSWIN1252) and never occurs
#   in UTF-8, so nothing is decoded unless the database's character set is UTF-8 (`_DECODABLE_CHARSETS`);
# - a NUL (CHR(0)) sixth byte, with small bytes after it, rounds onto the 5 bytes before it, in any character set. No
#   dictionary view says whether a column holds a NUL, so this one stays: a code list then names the code without it.
# The first and last values are also checked against LOW_VALUE and HIGH_VALUE. A sixth byte is never decoded: a seventh
# can hide below a step, so 'ABCDEF' and 'ABCDEF1' may round alike.
_ENCODED_BYTES = 15
_SIGNIFICANT_DIGITS = 15
_DECODED_MAX_BYTES = 5
_DECODABLE_CHARSETS = frozenset({"AL32UTF8", "UTF8"})

_TEXT_TYPES = frozenset({"VARCHAR2", "VARCHAR", "CHAR"})
_NUMBER_TYPES = frozenset({"NUMBER", "FLOAT", "INTEGER"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EstimatedColumn:
    column: str
    estimate: ColumnEstimate | None  # None: the dictionary holds no statistics for this column
    codes: list[str] = field(default_factory=list)
    declared_unique: bool = False


def estimated_tables(snapshot) -> frozenset[str]:
    """The tables of `snapshot` profiled from the source's statistics: nothing after the structural pass may read their
    rows. Read from their profile jobs, so a column without statistics in such a table counts too."""
    return frozenset(j.id.removeprefix("profile:") for j in snapshot.jobs
                     if j.kind == "profile" and j.status == "done" and j.detail == "estimated")


def estimate_table(adapter, table: TableInfo, key_columns: set[str] | None = None,
                   code_max_distinct: int = 25, charset: str | None = None) -> list[EstimatedColumn]:
    """Each column of `table` as the dictionary describes it. Reads the table's size, its columns' statistics and its
    declared keys, and a histogram only for a column that can be a code list, so a wide table costs one more read per
    candidate, not per column. Excludes from code lists what the exact profile excludes: key-like names, declared
    foreign-key children (`key_columns`), sensitive names, and a column with as many values as rows; and a column
    whose statistics were gathered at another time than the table's, whose histogram can lack codes added since.
    That time is `LAST_ANALYZED`, to the second, and nothing a reader can see names the gather: two gathers of the table
    within one second are one time here, a residual the deployment guide states.
    `charset` is the database's character set (`database_charset`), None when unknown (see `frequency_codes`)."""
    size = adapter.table_size(table.name)
    statistics = {s.column: s for s in adapter.column_statistics(table.name)}
    unique = adapter.unique_columns(table.name)
    keys = key_columns or set()
    columns: list[EstimatedColumn] = []
    for col in table.columns:
        st = statistics.get(col.name)
        if st is None:
            columns.append(EstimatedColumn(col.name, None, [], col.name in unique))
            continue
        estimate = ColumnEstimate(rows=size.rows, distinct=st.distinct, nulls=st.nulls,
                                  analyzed=st.analyzed or size.analyzed, sampled=_sampled(st, size.rows),
                                  stale=size.stale)
        codes: list[str] = []
        if _histogram_candidate(col, st, size, keys, code_max_distinct):
            codes = frequency_codes(st, size.rows, size.stale, adapter.histogram(table.name, col.name),
                                    col.data_type, code_max_distinct, charset)
        columns.append(EstimatedColumn(col.name, estimate, codes, col.name in unique))
    return columns


def _histogram_candidate(col, st: ColumnStatistics, size, keys: set[str], code_max_distinct: int) -> bool:
    """Whether `col` can carry a code list from its histogram, so the histogram is worth reading: what the exact
    profile excludes is excluded (key-like names, declared foreign-key children, sensitive names, as many values as
    rows), and statistics gathered at another time than the table's cannot give one."""
    return (not is_key_like(col.name) and col.name not in keys and not is_sensitive_name(col.name)
            and _kind(col.data_type) is not None
            and st.distinct is not None and size.rows is not None and st.distinct < size.rows
            and st.analyzed is not None and st.analyzed == size.analyzed
            and _may_be_complete(st, size.rows, size.stale, code_max_distinct))


def histogram_candidates(table: TableInfo, size, statistics: dict[str, ColumnStatistics],
                         key_columns: set[str] | None, code_max_distinct: int) -> list[str]:
    """The columns of `table` whose histogram `estimate_table` reads: one rule, so a prefetch reads exactly those."""
    keys = key_columns or set()
    return [col.name for col in table.columns
            if col.name in statistics and _histogram_candidate(col, statistics[col.name], size, keys, code_max_distinct)]


class PrefetchedDictionary:
    """An adapter whose dictionary reads come from one batched read of many tables (`prefetch_dictionary`): the four
    reads `estimate_table` makes are answered from it, and anything else, a table it does not hold included, is the
    adapter's."""

    def __init__(self, adapter, facts, histograms: dict) -> None:
        self._adapter, self._facts, self._histograms = adapter, facts, histograms

    def table_size(self, table: str):
        return self._facts.sizes[table] if table in self._facts.sizes else self._adapter.table_size(table)

    def column_statistics(self, table: str):
        return (self._facts.statistics[table] if table in self._facts.statistics
                else self._adapter.column_statistics(table))

    def unique_columns(self, table: str):
        return self._facts.unique[table] if table in self._facts.unique else self._adapter.unique_columns(table)

    def histogram(self, table: str, column: str):
        if (table, column) in self._histograms:
            return self._histograms[(table, column)]
        return self._adapter.histogram(table, column)

    def __getattr__(self, name: str):
        return getattr(self._adapter, name)


def prefetch_dictionary(adapter, tables: list[TableInfo], key_columns: dict[str, set[str]],
                        code_max_distinct: int):
    """`adapter` with the dictionary of `tables` read at once, where it can read it so (`dictionary`, `histograms`):
    statistics mode's 600 tables become a handful of statements instead of four or more each. An adapter without the
    batched reads is returned as it is, and read per table."""
    if not tables or not callable(getattr(adapter, "dictionary", None)):
        return adapter
    try:
        facts = adapter.dictionary([t.name for t in tables])
    except Exception as exc:  # noqa: BLE001 -- the per-table reads remain, so a failed batch costs time, not the run
        logger.warning("the batched dictionary read failed (%s); the dictionary is read per table", exc)
        return adapter
    wanted = []
    if code_max_distinct and callable(getattr(adapter, "histograms", None)):
        for t in tables:
            statistics = {st.column: st for st in facts.statistics[t.name]}
            wanted += [(t.name, c) for c in histogram_candidates(t, facts.sizes[t.name], statistics,
                                                                 key_columns.get(t.name), code_max_distinct)]
    try:
        histograms = adapter.histograms(wanted) if wanted else {}
    except Exception as exc:  # noqa: BLE001 -- as above: each histogram is then read on its own
        logger.warning("the batched histogram read failed (%s); histograms are read per table", exc)
        histograms = {}
    return PrefetchedDictionary(adapter, facts, histograms)


def frequency_codes(stats: ColumnStatistics, rows: int | None, stale: bool | None,
                    endpoints: list[HistogramEndpoint], data_type: str, code_max_distinct: int = 25,
                    charset: str | None = None) -> list[str]:
    """The column's codes, most rows first and then by value (as the exact profile orders them), or [] whenever the
    list cannot be shown complete and exact.

    Complete: a FREQUENCY histogram with one endpoint per distinct value, built from every non-null row, on statistics
    the database does not call stale (unknown counts as stale). Exact: a numeric column's values as the driver returns
    them; a character column's text where Oracle stores it, short of where it cuts text; otherwise, on a database whose
    `charset` is UTF-8, each value decoded from its ENDPOINT_VALUE, at most 5 printable ASCII bytes, re-encoding
    exactly, the first and last equal to the column's lowest and highest values."""
    if not _may_be_complete(stats, rows, stale, code_max_distinct) or len(endpoints) != stats.distinct:
        return []
    counts: list[int] = []
    previous = 0
    for ep in endpoints:
        if ep.number is None or ep.number <= previous:
            return []
        counts.append(ep.number - previous)
        previous = ep.number
    kind = _kind(data_type)
    if kind == "number":
        # The driver returns a FLOAT or a NUMBER with a scale above 0 as a float (2.0), its ENDPOINT_VALUE as an int (2).
        as_float = _base(data_type) == "FLOAT" or (stats.scale or 0) > 0
        values = [_number_code(ep, as_float) for ep in endpoints]
    elif kind == "text":
        values = _text_codes(endpoints, stats, decode=charset in _DECODABLE_CHARSETS)
    else:
        return []
    if values is None or any(v is None for v in values) or len(set(values)) != len(values):
        return []
    order = sorted(range(len(values)), key=lambda i: (-counts[i], i))
    return [values[i] for i in order]


def profile_report(snapshot) -> tuple[str | None, str | None]:
    """(summary, warning) for an operator after statistics mode: how many tables were profiled each way, and which
    estimated tables the source holds no statistics for, or statistics it marks stale or does not say are current.
    (None, None) when no table was estimated."""
    estimated = sorted(estimated_tables(snapshot))
    if not estimated:
        return None, None
    exact = sum(1 for j in snapshot.jobs if j.kind == "profile" and j.status == "done" and j.detail != "estimated")
    summary = f"profiled {exact} table(s) exactly, {len(estimated)} from Oracle's statistics"
    by_table: dict[str, list] = {}
    for column in snapshot.columns:
        by_table.setdefault(column.object_id, []).append(column.estimate)
    unread, partial, stale = [], [], []
    for table in estimated:
        everything = by_table.get(table, [])
        estimates = [e for e in everything if e is not None and e.rows is not None]
        if not estimates:
            unread.append(table)
            continue
        if len(estimates) < len(everything):
            partial.append(f"{table} ({len(everything) - len(estimates)} of {len(everything)} columns)")
        if any(e.stale is not False for e in estimates):
            analyzed = max((e.analyzed for e in estimates if e.analyzed), default=None)
            stale.append(f"{table} (as of {analyzed})" if analyzed else table)
    parts = []
    if unread:
        parts.append(f"{len(unread)} table(s) of source {snapshot.source_id!r} have no optimizer statistics, so no "
                     f"counts or code lists were read for them: {', '.join(unread)}.")
    if partial:
        parts.append(f"{len(partial)} table(s) of source {snapshot.source_id!r} have columns without statistics, so no "
                     f"counts or code lists were read for those columns: {', '.join(partial)}.")
    if stale:
        parts.append(f"{len(stale)} table(s) of source {snapshot.source_id!r} have statistics Oracle marks stale or "
                     f"does not say are current, so their counts may be out of date and no code list was read from "
                     f"them: {', '.join(stale)}.")
    if not parts:
        return summary, None
    parts.append("Gather their statistics (DBMS_STATS.GATHER_TABLE_STATS) and run enrich again, or name them in the "
                 "source's \"exact_tables\" to profile them in full.")
    return summary, " ".join(parts)


def unapplied_code_meanings(snapshot, dictionary) -> list[str]:
    """Columns of estimated tables for which the operator's dictionary gives code meanings that `apply_dictionary` will
    not apply, since it attaches a meaning only to a code in the column's list, and an estimated column has a list only
    where a histogram allowed one. Each as "TABLE.COLUMN (n of m codes)". Exactly profiled tables are not looked at."""
    estimated = estimated_tables(snapshot)
    unapplied = []
    for column in snapshot.columns:
        entry = dictionary.columns.get(column.id) if column.object_id in estimated else None
        if entry is None or not entry.codes:
            continue
        listed = {cv.code for cv in column.coded_values}
        missing = [code for code in entry.codes if code not in listed]
        if missing:
            unapplied.append(f"{column.id} ({len(missing)} of {len(entry.codes)} codes)")
    return unapplied


def _may_be_complete(stats: ColumnStatistics, rows: int | None, stale: bool | None, code_max_distinct: int) -> bool:
    return (stats.histogram == "FREQUENCY" and stale is False
            and stats.distinct is not None and 0 < stats.distinct <= code_max_distinct
            and stats.buckets == stats.distinct
            and rows is not None and stats.nulls is not None and stats.sample_size == rows - stats.nulls)


def _sampled(stats: ColumnStatistics, rows: int | None) -> bool | None:
    if stats.sample_size is None or rows is None or stats.nulls is None:
        return None
    return stats.sample_size < rows - stats.nulls


def _base(data_type: str) -> str:
    return (data_type or "").upper().split("(")[0].strip()


def _kind(data_type: str) -> str | None:
    base = _base(data_type)
    if base in _NUMBER_TYPES:
        return "number"
    if base in _TEXT_TYPES:
        return "text"
    return None  # a date's endpoint is a day number; NVARCHAR2/NCHAR are encoded otherwise; the rest are not codes


def _number_code(ep: HistogramEndpoint, as_float: bool = False) -> str | None:
    value = ep.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if as_float:
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        code = str(value)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        code = str(int(value)) if value.is_integer() else str(value)
    else:
        code = str(value)
    if ep.actual is not None:
        try:
            if Decimal(ep.actual) != Decimal(code):
                return None
        except InvalidOperation:
            return None
    return code


def _text_codes(endpoints: list[HistogramEndpoint], stats: ColumnStatistics,
                decode: bool) -> list[str | None] | None:
    stored = [ep.actual is not None or ep.actual_raw is not None for ep in endpoints]
    if all(stored):
        return [_stored_text(ep) for ep in endpoints]
    if any(stored) or not decode:
        return None  # a mix was never seen; read as neither
    decoded = [_decoded(ep.value) for ep in endpoints]
    if any(d is None for d in decoded) or stats.low is None or stats.high is None:
        return None
    if decoded[0] != stats.low or decoded[-1] != stats.high:
        return None
    return [d.decode("ascii") for d in decoded]


def _stored_text(ep: HistogramEndpoint) -> str | None:
    if ep.actual is not None:
        text = ep.actual
    elif _printable_ascii(ep.actual_raw):
        text = ep.actual_raw.decode("ascii")
    else:
        return None
    size = max(len(text.encode("utf-8")), len(ep.actual_raw or b""))
    limit = _TEXT_BYTES_KEPT if text.isascii() else _NON_ASCII_TEXT_MAY_BE_CUT_AT
    return None if size >= limit else text


def _decoded(value) -> bytes | None:
    """The shortest text of at most 5 printable ASCII bytes whose encoding is exactly `value`, or None."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    for length in range(1, _DECODED_MAX_BYTES + 1):
        unit = 256 ** (_ENCODED_BYTES - length)
        prefix = (value + unit // 2) // unit
        if prefix >= 256 ** length:
            return None
        text = prefix.to_bytes(length, "big")
        if _encoded(text) == value:
            return text if _printable_ascii(text) else None
    return None


def _encoded(text: bytes) -> int:
    whole = int.from_bytes(text[:_ENCODED_BYTES].ljust(_ENCODED_BYTES, b"\0"), "big")
    digits = len(str(whole))
    if digits <= _SIGNIFICANT_DIGITS:
        return whole
    step = 10 ** (digits - _SIGNIFICANT_DIGITS)
    kept, rest = divmod(whole, step)
    return (kept + (1 if 2 * rest >= step else 0)) * step


def _printable_ascii(raw: bytes | None) -> bool:
    return bool(raw) and all(0x20 <= b <= 0x7E for b in raw)
