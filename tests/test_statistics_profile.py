"""A table's profile from Oracle's dictionary, reading none of its rows (`mnemiq.enrichment.statistics`).

Counts land in `Column.estimate`, never in the exact counts; a key is unique only where the table declares it; a code
list is given only where a frequency histogram covers every row and every value can be read back exactly, and is
otherwise empty, never partial or wrong.

The endpoint values below are the ones Oracle 23.26.2 wrote for these codes (measured 2026-11-20 on a synthetic table:
`DBMS_STATS.GATHER_TABLE_STATS`, SIZE 254 and SIZE AUTO alike). On a VARCHAR2 column of short codes Oracle stores no
text for the endpoints, only `ENDPOINT_VALUE`: the value's first 15 bytes as an integer, rounded to 15 significant
digits. python-oracledb returns a NUMBER as an int when it is integral and a float otherwise, for a column's values and
for `ENDPOINT_VALUE` alike, so a numeric code is written as the exact profile writes the value it reads.
"""
from __future__ import annotations

import pytest

from mnemiq.adapters.dictionary import ColumnStatistics, HistogramEndpoint, TableSize
from mnemiq.catalog import ColumnInfo, TableInfo
from mnemiq.contract.semantic import ColumnEstimate
from mnemiq.enrichment.statistics import EstimatedColumn, estimate_table, frequency_codes

E21 = 10**21
# Oracle's ENDPOINT_VALUE for each code, as measured.
Y, N = 462114420409600 * E21, 404999154965717 * E21
A = 337499295804764 * E21
ON, OFF = 411773479773336 * E21, 411616766467883 * E21
A1_TO_A9 = [v * E21 for v in (338493133875343, 338513416284946, 338533698694550, 338553981104154, 338574263513757,
                              338594545923361, 338614828332965, 338635110742568, 338655393152172)]
ABCDEF, ABCDEG = 338843264254220 * E21, 338843264254225 * E21
E_ACUTE = 1015925614637310 * E21  # 'é' in AL32UTF8


def _stats(column="STATUS", distinct=2, nulls=0, histogram="FREQUENCY", buckets=None, sample_size=None,
           low=None, high=None, rows=10_000):
    return ColumnStatistics(column=column, distinct=distinct, nulls=nulls, histogram=histogram,
                            buckets=distinct if buckets is None else buckets,
                            sample_size=rows - nulls if sample_size is None else sample_size,
                            analyzed="2026-11-01", low=low, high=high)


def _ep(number, value, actual=None, raw=None):
    return HistogramEndpoint(number=number, value=value, actual=actual, actual_raw=raw)


YN = [_ep(3000, N), _ep(10_000, Y)]


def _codes(stats=None, endpoints=YN, data_type="VARCHAR2", rows=10_000, stale=False, **kw):
    stats = stats or _stats(low=b"N", high=b"Y")
    return frequency_codes(stats, rows, stale, endpoints, data_type, **kw)


# --- frequency_codes: the code list, only where it is complete and exact --------------------------

def test_frequency_codes_from_a_complete_histogram():
    """Ordered as the exact profile orders them: most rows first, then by value."""
    endpoints = [_ep(1000, None, "A", b"A"), _ep(1700, None, "B", b"B"), _ep(10_000, None, "C", b"C")]
    assert _codes(_stats(distinct=3), endpoints) == ["C", "A", "B"]


def test_frequency_codes_decode_short_codes_from_the_endpoint_value():
    """No text on the endpoints, as Oracle writes them for short VARCHAR2 codes: decoded, and checked against the
    column's lowest and highest values."""
    assert _codes() == ["Y", "N"]
    nine = [_ep(1112 + 1111 * i if i < 8 else 10_000, v) for i, v in enumerate(A1_TO_A9)]
    assert _codes(_stats(distinct=9, low=b"A1", high=b"A9"), nine) == [f"A{i}" for i in range(1, 10)]
    assert _codes(_stats(distinct=1, low=b"A", high=b"A", rows=10_000), [_ep(10_000, A)]) == ["A"]
    assert _codes(_stats(distinct=2, low=b"OFF", high=b"ON", rows=3000), [_ep(1500, OFF), _ep(3000, ON)],
                  rows=3000) == ["OFF", "ON"]


def test_frequency_codes_refuse_a_decode_longer_than_five_bytes():
    """A sixth byte can be decoded, but the seventh cannot be seen: 'ABCDEF' and 'ABCDEF1' may round alike."""
    endpoints = [_ep(3334, ABCDEF), _ep(10_000, ABCDEG)]
    assert _codes(_stats(low=b"ABCDEF", high=b"ABCDEG"), endpoints) == []


def test_frequency_codes_refuse_a_decode_the_low_or_high_value_contradicts():
    assert _codes(_stats(low=b"M", high=b"Y")) == []
    assert _codes(_stats(low=b"N", high=b"YES")) == []


def test_frequency_codes_refuse_a_decode_without_the_low_and_high_values():
    assert _codes(_stats(low=None, high=b"Y")) == []


def test_frequency_codes_refuse_a_decode_that_does_not_re_encode():
    """A value one step off what any string encodes to is not read as the nearest string."""
    assert _codes(endpoints=[_ep(3000, N + E21), _ep(10_000, Y)]) == []


def test_frequency_codes_refuse_non_ascii_text_they_would_decode():
    assert _codes(_stats(distinct=1, low="é".encode(), high="é".encode()), [_ep(10_000, E_ACUTE)]) == []


def test_frequency_codes_read_the_text_oracle_stores():
    """Where Oracle stores the text (a CHAR column, or values sharing their leading bytes), it is read as stored."""
    endpoints = [_ep(3334, 456922123551065 * E21, "X  ", b"X  "), _ep(10_000, 463919554864325 * E21, "YY ", b"YY ")]
    assert _codes(_stats(), endpoints, data_type="CHAR") == ["YY ", "X  "]
    longer = [_ep(2000, 1, "LONGCODE_ALPHA_01", b"LONGCODE_ALPHA_01"), _ep(10_000, 1, "LONGCODE_ALPHA_02", None)]
    assert _codes(_stats(), longer) == ["LONGCODE_ALPHA_02", "LONGCODE_ALPHA_01"]


def test_frequency_codes_read_raw_bytes_when_actual_is_empty():
    assert _codes(endpoints=[_ep(3000, N, None, b"N"), _ep(10_000, Y, None, b"Y")]) == ["Y", "N"]


def test_frequency_codes_refuse_text_oracle_may_have_cut():
    """Oracle keeps 64 bytes of an endpoint's text; at 64 the value may be longer."""
    cut = "X" * 64
    assert _codes(_stats(distinct=1), [_ep(10_000, 1, cut, cut.encode())]) == []
    assert _codes(_stats(distinct=1), [_ep(10_000, 1, "X" * 63, None)]) == ["X" * 63]


def test_frequency_codes_refuse_values_collapsed_into_fewer_endpoints():
    """Three 80-byte values sharing 64 bytes: Oracle writes one endpoint for three distinct values (measured)."""
    stats = _stats(distinct=3, buckets=1)
    assert _codes(stats, [_ep(3000, 1, "X" * 64, None)], rows=3000) == []


def test_frequency_codes_refuse_a_sampled_histogram():
    assert _codes(_stats(sample_size=5000, low=b"N", high=b"Y")) == []


def test_frequency_codes_refuse_stale_statistics():
    assert _codes(stale=True) == []


def test_frequency_codes_refuse_unknown_staleness():
    assert _codes(stale=None) == []


@pytest.mark.parametrize("kind", ["TOP-FREQUENCY", "HYBRID", "HEIGHT BALANCED", "NONE"])
def test_frequency_codes_refuse_top_frequency_and_hybrid(kind):
    assert _codes(_stats(histogram=kind, low=b"N", high=b"Y")) == []


def test_frequency_codes_refuse_more_than_25():
    endpoints = [_ep(i + 1, None, f"C{i:02d}", None) for i in range(26)]
    assert _codes(_stats(distinct=26, rows=26), endpoints, rows=26) == []
    assert len(_codes(_stats(distinct=25, rows=25), endpoints[:25], rows=25)) == 25


def test_frequency_codes_refuse_an_endpoint_count_unlike_the_distinct_count():
    assert _codes(_stats(distinct=3, low=b"N", high=b"Y")) == []


@pytest.mark.parametrize("data_type", ["DATE", "TIMESTAMP(6)", "NVARCHAR2", "NCHAR", "RAW", "BINARY_DOUBLE", "CLOB"])
def test_frequency_codes_refuse_a_date_column(data_type):
    """A date's endpoint is a day number; a national character column is encoded otherwise; the rest are not codes."""
    endpoints = [_ep(3334, 2461042, None, None), _ep(10_000, 2461043, None, None)]
    assert _codes(_stats(), endpoints, data_type=data_type) == []


def test_frequency_codes_write_numbers_as_the_exact_profile_does():
    """The exact profile writes `str(value)` of what the driver returns: an int for an integral NUMBER, else a float."""
    endpoints = [_ep(3334, 1.5, "1.5", b"\xc1\x02\x33"), _ep(6667, 2.5, "2.5", None), _ep(10_000, 10, "10", None)]
    assert _codes(_stats(distinct=3), endpoints, data_type="NUMBER") == ["1.5", "2.5", "10"]
    assert _codes(_stats(distinct=2), [_ep(5000, 1.0, None, None), _ep(10_000, 2, None, None)],
                  data_type="NUMBER") == ["1", "2"]


def test_frequency_codes_refuse_a_number_its_text_contradicts():
    endpoints = [_ep(5000, 1, "1", None), _ep(10_000, 2, "3", None)]
    assert _codes(_stats(), endpoints, data_type="NUMBER") == []


# --- estimate_table: the whole table, from the dictionary alone -----------------------------------

class _Source:
    """An Oracle source as the statistics profile sees it. Reading any table row fails the test."""

    dialect = "oracle"

    def __init__(self, size=None, columns=(), histograms=None, unique=frozenset()):
        self.size = size or TableSize(rows=1000, avg_row_len=40, analyzed="2026-11-01", stale=False)
        self.columns = list(columns)
        self.histograms = histograms or {}
        self.unique = unique
        self.histograms_read: list[str] = []

    def table_size(self, table):
        return self.size

    def column_statistics(self, table):
        return self.columns

    def histogram(self, table, column):
        self.histograms_read.append(column)
        return self.histograms.get(column, [])

    def unique_columns(self, table):
        return self.unique

    def execute(self, sql):
        raise AssertionError(f"the statistics profile read table data: {sql}")


ORDERS = TableInfo(name="ORDERS", columns=[ColumnInfo("ID", "NUMBER"), ColumnInfo("STATUS", "VARCHAR2"),
                                           ColumnInfo("NOTE", "VARCHAR2")])


def _orders_source(**kw):
    return _Source(columns=[
        _stats("ID", distinct=1000, histogram="NONE", rows=1000),
        _stats("STATUS", distinct=2, rows=1000, low=b"N", high=b"Y"),
    ], histograms={"STATUS": [_ep(300, N), _ep(1000, Y)]}, **kw)


def test_estimate_table_reads_no_table_data():
    assert len(estimate_table(_orders_source(), ORDERS)) == 3


def test_counts_land_in_the_estimate_only():
    col = estimate_table(_orders_source(), ORDERS)[0]
    assert col == EstimatedColumn(column="ID", estimate=ColumnEstimate(rows=1000, distinct=1000, nulls=0,
                                                                       analyzed="2026-11-01", sampled=False,
                                                                       stale=False),
                                  codes=[], declared_unique=False)


def test_a_sampled_column_says_so():
    source = _Source(columns=[_stats("ID", distinct=990, histogram="NONE", rows=1000, sample_size=200)])
    assert estimate_table(source, ORDERS)[0].estimate.sampled is True


def test_a_column_without_statistics_has_no_estimate():
    note = estimate_table(_orders_source(), ORDERS)[2]
    assert (note.column, note.estimate, note.codes) == ("NOTE", None, [])


def test_staleness_comes_from_the_table():
    source = _orders_source(size=TableSize(rows=1000, avg_row_len=40, analyzed="2026-11-01", stale=None))
    cols = estimate_table(source, ORDERS)
    assert cols[0].estimate.stale is None and cols[1].codes == []


def test_declared_unique_from_unique_columns():
    cols = estimate_table(_orders_source(unique=frozenset({"ID"})), ORDERS)
    assert [c.declared_unique for c in cols] == [True, False, False]


def test_codes_from_the_histogram_of_a_code_column():
    source = _orders_source()
    assert estimate_table(source, ORDERS)[1].codes == ["Y", "N"]
    assert source.histograms_read == ["STATUS"], "a histogram is read only for a column that can be a code list"


def test_no_codes_for_a_key_like_fk_child_or_sensitive_column():
    table = TableInfo(name="ORDERS", columns=[ColumnInfo("CUSTOMER_ID", "VARCHAR2"), ColumnInfo("REGION", "VARCHAR2"),
                                              ColumnInfo("CUSTOMER_NAME", "VARCHAR2"), ColumnInfo("STATUS", "VARCHAR2")])
    source = _Source(columns=[_stats(c, distinct=2, rows=1000, low=b"N", high=b"Y")
                              for c in ("CUSTOMER_ID", "REGION", "CUSTOMER_NAME", "STATUS")],
                     histograms={c: [_ep(300, N), _ep(1000, Y)]
                                 for c in ("CUSTOMER_ID", "REGION", "CUSTOMER_NAME", "STATUS")})
    cols = estimate_table(source, table, key_columns={"REGION"})
    assert [c.codes for c in cols] == [[], [], [], ["Y", "N"]]
    assert source.histograms_read == ["STATUS"]


def test_no_codes_where_every_row_is_distinct():
    """As the exact profile: a column with as many values as rows is not a vocabulary."""
    source = _Source(size=TableSize(rows=2, avg_row_len=40, analyzed="2026-11-01", stale=False),
                     columns=[_stats("STATUS", distinct=2, rows=2, low=b"N", high=b"Y")],
                     histograms={"STATUS": [_ep(1, N), _ep(2, Y)]})
    table = TableInfo(name="ORDERS", columns=[ColumnInfo("STATUS", "VARCHAR2")])
    assert estimate_table(source, table)[0].codes == []


def test_a_histogram_with_fewer_buckets_than_values_is_not_read():
    """Values Oracle collapsed into fewer endpoints can never give a complete list, so their histogram is not read."""
    source = _Source(columns=[_stats("STATUS", distinct=3, buckets=1, rows=1000)],
                     histograms={"STATUS": [_ep(1000, 1, "X" * 64, None)]})
    table = TableInfo(name="ORDERS", columns=[ColumnInfo("STATUS", "VARCHAR2")])
    assert estimate_table(source, table)[0].codes == [] and source.histograms_read == []
