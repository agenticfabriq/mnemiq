"""What the Oracle dictionary says about a table, read with no table data: `OracleAdapter.table_size`,
`column_statistics`, `histogram` and `unique_columns`.

The dictionary is stubbed (`_rows` answered by SQL shape, as `test_oracle_resolve_called_functions.py` does); the live
reads against Oracle are `test_oracle_adapter.py::test_table_size_reads_the_statistics_oracle_gathered` and
`test_dictionary_reads_live`.
"""
from __future__ import annotations

import datetime

import pytest

from mnemiq.adapters.dictionary import ColumnStatistics, HistogramEndpoint, TableSize
from mnemiq.adapters.oracle import OracleAdapter
from mnemiq.adapters.scoped import TableScopedAdapter, TablesNotFound


class _Statistics:
    """`ALL_TAB_STATISTICS` as rows of (owner, table, object_type, num_rows, avg_row_len, last_analyzed, stale)."""

    def __init__(self, rows):
        self.table = list(rows)
        self.sql: list[str] = []
        self.binds: list[dict] = []

    def rows(self, sql, **binds):
        self.sql.append(sql)
        self.binds.append(binds)
        # The read must keep to the table's own row: a partitioned table has one per partition too.
        object_type = "TABLE" if "object_type = 'TABLE'" in sql.lower().replace('"', "'") else None
        return [(n, a, la, st) for o, t, ot, n, a, la, st in self.table
                if o == binds["owner"] and t == binds["table_name"] and (object_type is None or ot == object_type)]


def _adapter(stats: _Statistics, schema: str = "APP") -> OracleAdapter:
    a = OracleAdapter.__new__(OracleAdapter)
    a._schema = schema
    a._rows = stats.rows
    return a


ANALYZED = datetime.datetime(2026, 11, 1, 6, 0, 15)


def test_table_size_reads_the_table_row_only():
    stats = _Statistics([("APP", "ORDERS", "TABLE", 1000, 80, ANALYZED, "NO"),
                         ("APP", "ORDERS", "PARTITION", 400, 80, ANALYZED, "NO")])
    assert _adapter(stats).table_size("ORDERS") == TableSize(rows=1000, avg_row_len=80,
                                                             analyzed="2026-11-01T06:00:15", stale=False)
    assert "object_type = 'table'" in stats.sql[-1].lower()


def test_table_size_of_an_unanalyzed_table_is_all_none():
    stats = _Statistics([("APP", "ORDERS", "TABLE", None, None, None, None)])
    assert _adapter(stats).table_size("ORDERS") == TableSize(rows=None, avg_row_len=None, analyzed=None, stale=None)


def test_table_size_of_a_table_the_dictionary_does_not_show_is_all_none():
    assert _adapter(_Statistics([])).table_size("ORDERS") == TableSize(None, None, None, None)


@pytest.mark.parametrize(("flag", "stale"), [("YES", True), ("NO", False), (None, None), ("", None)])
def test_table_size_maps_staleness_three_ways(flag, stale):
    """Unknown is not "not stale": a NULL STALE_STATS on an analyzed table stays None."""
    stats = _Statistics([("APP", "ORDERS", "TABLE", 1000, 80, ANALYZED, flag)])
    assert _adapter(stats).table_size("ORDERS").stale is stale


def test_table_size_binds_the_owner_and_the_table():
    stats = _Statistics([("APP", "Orders", "TABLE", 10, 8, ANALYZED, "NO")])
    _adapter(stats).table_size("Orders")
    assert stats.binds[-1] == {"owner": "APP", "table_name": "Orders"}
    assert "Orders" not in stats.sql[-1]


class _Inner:
    def __init__(self):
        self.asked: list[str] = []

    def list_columns(self):
        return [("ORDERS", "ID", "NUMBER"), ("PAYROLL", "ID", "NUMBER")]

    def table_size(self, table):
        self.asked.append(table)
        return TableSize(10, 8, "2026-11-01", False)

    def column_statistics(self, table):
        self.asked.append(table)
        return []

    def histogram(self, table, column):
        self.asked.append(table)
        return []

    def unique_columns(self, table):
        self.asked.append(table)
        return frozenset()


def test_scoped_adapter_reads_a_table_in_scope():
    inner = _Inner()
    assert TableScopedAdapter(inner, ("orders",), "src").table_size("ORDERS").rows == 10
    assert inner.asked == ["ORDERS"]


def test_scoped_adapter_refuses_a_table_out_of_scope():
    inner = _Inner()
    with pytest.raises(TablesNotFound, match="PAYROLL"):
        TableScopedAdapter(inner, ("orders",), "src").table_size("PAYROLL")
    assert inner.asked == []


# --- a column's statistics, its histogram, and the keys the table declares ------------------------

class _Dictionary:
    """The four dictionary views the statistics profile reads, keyed by (owner, table), answered by which view the SQL
    names. Constraint and index rows come back unfiltered, so the adapter's own filter is what the tests read."""

    def __init__(self, columns=(), histograms=None, constraints=(), indexes=()):
        self.columns = list(columns)          # (column, ndv, nulls, histogram, buckets, sample, analyzed, low, high)
        self.histograms = histograms or {}    # column -> [(number, value, actual, raw)]
        self.constraints = list(constraints)  # (name, type, status, validated, column)
        self.indexes = list(indexes)          # (name, uniqueness, status, index_type, column)
        self.sql: list[str] = []
        self.binds: list[dict] = []

    def rows(self, sql, **binds):
        self.sql.append(sql)
        self.binds.append(binds)
        if (binds.get("owner"), binds.get("table_name")) != ("APP", "ORDERS"):
            return []
        text = sql.lower()
        if "all_tab_col_statistics" in text:
            return list(self.columns)
        if "all_tab_histograms" in text:
            return list(self.histograms.get(binds["column_name"], []))
        if "all_constraints" in text:
            return list(self.constraints)
        if "all_indexes" in text:
            return list(self.indexes)
        raise AssertionError(f"unexpected dictionary read: {sql}")


def _over(dictionary: _Dictionary) -> OracleAdapter:
    a = OracleAdapter.__new__(OracleAdapter)
    a._schema = "APP"
    a._rows = dictionary.rows
    return a


def test_column_statistics_maps_each_column():
    d = _Dictionary(columns=[
        ("STATUS", 3, 0, "FREQUENCY", 3, 1000, ANALYZED, b"A", b"C", None),
        ("NOTE", None, None, "NONE", None, None, None, None, None, None),
        ("RATE", 4, 0, "NONE", None, 1000, ANALYZED, None, None, 2),
    ])
    assert _over(d).column_statistics("ORDERS") == [
        ColumnStatistics(column="STATUS", distinct=3, nulls=0, histogram="FREQUENCY", buckets=3, sample_size=1000,
                         analyzed="2026-11-01T06:00:15", low=b"A", high=b"C", scale=None),
        ColumnStatistics(column="NOTE", distinct=None, nulls=None, histogram="NONE", buckets=None, sample_size=None,
                         analyzed=None, low=None, high=None, scale=None),
        ColumnStatistics(column="RATE", distinct=4, nulls=0, histogram="NONE", buckets=None, sample_size=1000,
                         analyzed="2026-11-01T06:00:15", low=None, high=None, scale=2),
    ]
    assert "all_tab_columns" in d.sql[-1].lower() and "data_scale" in d.sql[-1].lower()


def test_analyzed_keeps_the_time_of_day():
    """A later gather on the same day that names other columns moves the table's time and leaves a column's histogram
    as it was, still marked not stale (measured on 23.26): only the time of day tells the two apart."""
    later = datetime.datetime(2026, 11, 1, 18, 0, 0)
    table = _adapter(_Statistics([("APP", "ORDERS", "TABLE", 1000, 80, later, "NO")])).table_size("ORDERS")
    column = _over(_Dictionary(columns=[("STATUS", 3, 0, "FREQUENCY", 3, 1000, ANALYZED, b"A", b"C", None)]))
    assert (table.analyzed, column.column_statistics("ORDERS")[0].analyzed) == ("2026-11-01T18:00:00",
                                                                                "2026-11-01T06:00:15")


def test_database_charset_reads_the_database_character_set():
    asked = []

    def rows(sql, **binds):
        asked.append((sql.lower(), binds))
        return [("WE8MSWIN1252",)]

    a = OracleAdapter.__new__(OracleAdapter)
    a._rows = rows
    assert a.database_charset() == "WE8MSWIN1252"
    sql, binds = asked[0]
    assert "nls_database_parameters" in sql and "nls_characterset" in sql and binds == {}


def test_database_charset_the_dictionary_does_not_show_is_none():
    a = OracleAdapter.__new__(OracleAdapter)
    a._rows = lambda sql, **binds: []
    assert a.database_charset() is None


def test_column_statistics_of_a_table_the_dictionary_does_not_show_is_empty():
    assert _over(_Dictionary()).column_statistics("PAYMENTS") == []


def test_histogram_endpoints_in_order_with_raw_bytes():
    d = _Dictionary(histograms={"STATUS": [(3000, 404999154965717 * 10**21, None, None),
                                           (10000, 462114420409600 * 10**21, "Y", b"Y")]})
    assert _over(d).histogram("ORDERS", "STATUS") == [
        HistogramEndpoint(number=3000, value=404999154965717 * 10**21, actual=None, actual_raw=None),
        HistogramEndpoint(number=10000, value=462114420409600 * 10**21, actual="Y", actual_raw=b"Y"),
    ]
    assert "order by endpoint_number" in d.sql[-1].lower()
    assert d.binds[-1] == {"owner": "APP", "table_name": "ORDERS", "column_name": "STATUS"}


def test_unique_columns_reads_only_enabled_validated_single_column_keys():
    d = _Dictionary(
        constraints=[("PK_ORDERS", "P", "ENABLED", "VALIDATED", "ID"),
                     ("UK_AB", "U", "ENABLED", "VALIDATED", "A"), ("UK_AB", "U", "ENABLED", "VALIDATED", "B"),
                     ("UK_C", "U", "DISABLED", "NOT VALIDATED", "C"),
                     ("UK_F", "U", "ENABLED", "NOT VALIDATED", "F")],
        indexes=[("IX_D", "UNIQUE", "VALID", "NORMAL", "D"),
                 ("IX_E", "UNIQUE", "UNUSABLE", "NORMAL", "E"),
                 ("IX_G", "NONUNIQUE", "VALID", "NORMAL", "G"),
                 ("IX_H", "UNIQUE", "VALID", "FUNCTION-BASED NORMAL", "SYS_NC00009$")],
    )
    assert _over(d).unique_columns("ORDERS") == frozenset({"ID", "D"})


def test_every_dictionary_read_binds_the_owner_and_the_table():
    d = _Dictionary()
    a = _over(d)
    a.column_statistics("Orders")
    a.histogram("Orders", "Status")
    a.unique_columns("Orders")
    assert all(b["owner"] == "APP" and b["table_name"] == "Orders" for b in d.binds)
    assert all("Orders" not in sql and "Status" not in sql for sql in d.sql)


@pytest.mark.parametrize(("method", "args"), [("column_statistics", ()), ("histogram", ("ID",)),
                                              ("unique_columns", ())])
def test_scoped_adapter_refuses_every_dictionary_read_out_of_scope(method, args):
    inner = _Inner()
    with pytest.raises(TablesNotFound, match="PAYROLL"):
        getattr(TableScopedAdapter(inner, ("orders",), "src"), method)("PAYROLL", *args)
    assert inner.asked == []


@pytest.mark.parametrize(("method", "args"), [("column_statistics", ()), ("histogram", ("ID",)),
                                              ("unique_columns", ())])
def test_scoped_adapter_passes_every_dictionary_read_in_scope(method, args):
    inner = _Inner()
    getattr(TableScopedAdapter(inner, ("orders",), "src"), method)("ORDERS", *args)
    assert inner.asked == ["ORDERS"]


# -- the dictionary for many tables at once ----------------------------------------------------------------------------
# Statistics mode read four dictionary statements per table, about 2,400 for 600 tables. `dictionary` reads each view
# once for up to DICTIONARY_BATCH tables, and `histograms` the endpoints of many columns at once; each must give what
# the per-table reads give.


class _Many:
    """The dictionary of several tables, answering the per-table reads and the batched ones by SQL shape."""

    def __init__(self, tables):
        self.tables = tables   # name -> {"size": row | None, "stats": [rows], "constraints": [rows], "indexes": [rows],
        self.sql: list[str] = []   # "histograms": {column: [rows]}}
        self.binds: list[dict] = []

    def rows(self, sql, **binds):
        self.sql.append(sql)
        self.binds.append(binds)
        s = sql.lower()
        batched = "table_name in (" in s or ") in ((" in s   # a per-table read has `constraint_type IN (` too
        if "all_tab_histograms" in s:
            pairs = ([(binds[f"t{i}"], binds[f"c{i}"]) for i in range(len(binds) // 2)] if batched
                     else [(binds["table_name"], binds["column_name"])])
            return [((t, c) if batched else ()) + tuple(row)
                    for t, c in pairs for row in self.tables.get(t, {}).get("histograms", {}).get(c, [])]
        names = [v for k, v in sorted(binds.items()) if k[0] == "t" and k[1:].isdigit()] if batched else [binds["table_name"]]
        key = ("size" if "all_tab_statistics" in s else "stats" if "all_tab_col_statistics" in s
               else "constraints" if "all_constraints" in s else "indexes")
        out = []
        for name in names:
            got = self.tables.get(name, {}).get(key)
            for row in ([got] if key == "size" and got else [] if key == "size" else got or []):
                out.append(((name,) if batched else ()) + tuple(row))
        return out


def _many_adapter(many: _Many) -> OracleAdapter:
    a = OracleAdapter.__new__(OracleAdapter)
    a._schema = "APP"
    a._rows = many.rows
    return a


_TWO = {
    "ORDERS": {"size": (1000, 40, ANALYZED, "NO"),
               "stats": [("ID", 1000, 0, "NONE", 1, 1000, ANALYZED, b"\x01", b"\x02", 0),
                         ("STATUS", 3, 5, "FREQUENCY", 3, 995, ANALYZED, b"A", b"C", None)],
               "constraints": [("PK_O", "P", "ENABLED", "VALIDATED", "ID")],
               "indexes": [("PK_O", "UNIQUE", "VALID", "NORMAL", "ID"), ("UX_FN", "UNIQUE", "VALID", "FUNCTION-BASED NORMAL", "STATUS")],
               "histograms": {"STATUS": [(2, 1.0, "A", None), (5, 2.0, None, b"C")]}},
    "NEVER_ANALYZED": {"size": None, "stats": [], "constraints": [], "indexes": []},
}


def test_dictionary_matches_the_per_table_reads():
    a = _many_adapter(_Many(_TWO))
    facts = a.dictionary(["ORDERS", "NEVER_ANALYZED"])
    for table in _TWO:
        assert facts.sizes[table] == a.table_size(table)
        assert facts.statistics[table] == a.column_statistics(table)
        assert facts.unique[table] == a.unique_columns(table)
    assert facts.unique["ORDERS"] == frozenset({"ID"})


def test_dictionary_of_a_table_the_dictionary_does_not_show_is_empty():
    facts = _many_adapter(_Many({})).dictionary(["MISSING"])
    assert facts.sizes["MISSING"] == TableSize(rows=None, avg_row_len=None, analyzed=None, stale=None)
    assert (facts.statistics["MISSING"], facts.unique["MISSING"]) == ([], frozenset())


def test_dictionary_splits_more_than_1000_tables():
    # Oracle refuses an IN-list of more than 1,000 items (ORA-01795).
    many = _Many({})
    _many_adapter(many).dictionary([f"T{i}" for i in range(1001)])
    per_view = {}
    for sql, binds in zip(many.sql, many.binds):
        view = next(v for v in ("all_tab_statistics", "all_tab_col_statistics", "all_constraints", "all_indexes")
                    if v in sql.lower())
        per_view.setdefault(view, []).append(len([k for k in binds if k != "owner"]))
    assert {view: sizes for view, sizes in per_view.items()} == {
        v: [500, 500, 1] for v in ("all_tab_statistics", "all_tab_col_statistics", "all_constraints", "all_indexes")}
    assert all(b["owner"] == "APP" for b in many.binds)


def test_dictionary_of_no_tables_reads_nothing():
    many = _Many({})
    facts = _many_adapter(many).dictionary([])
    assert (facts.sizes, facts.statistics, facts.unique, many.sql) == ({}, {}, {}, [])


def test_histograms_match_the_per_column_reads():
    a = _many_adapter(_Many(_TWO))
    got = a.histograms([("ORDERS", "STATUS"), ("ORDERS", "ID"), ("NEVER_ANALYZED", "X")])
    for table, column in (("ORDERS", "STATUS"), ("ORDERS", "ID"), ("NEVER_ANALYZED", "X")):
        assert got[(table, column)] == a.histogram(table, column)
    assert [e.actual for e in got[("ORDERS", "STATUS")]] == ["A", None]


def test_600_tables_cost_a_handful_of_dictionary_statements():
    many = _Many({})
    _many_adapter(many).dictionary([f"T{i}" for i in range(600)])
    assert len(many.sql) == 8, "four views, two batches each, against 2,400 per-table reads"


def test_table_sizes_reads_one_view_in_batches():
    many = _Many(_TWO)
    a = _many_adapter(many)
    sizes = a.table_sizes(["ORDERS", "NEVER_ANALYZED"])
    assert sizes == {t: a.table_size(t) for t in ("ORDERS", "NEVER_ANALYZED")}
    assert "all_tab_statistics" in many.sql[0].lower() and len(many.sql) == 3  # one batched, two per-table to compare


def test_the_table_scope_refuses_a_batched_read_of_a_table_out_of_scope():
    # The scope guards each per-table dictionary read; the batched reads must not hand an unlisted table's
    # statistics, sizes or histogram values through (Codex's review of #277).
    class Inner:
        def __init__(self):
            self.asked = []

        def dictionary(self, tables):
            self.asked.append(("dictionary", list(tables)))

        def table_sizes(self, tables):
            self.asked.append(("table_sizes", list(tables)))

        def histograms(self, columns):
            self.asked.append(("histograms", list(columns)))

    inner = Inner()
    scoped = TableScopedAdapter(inner, ("ORDERS", "dim_*"), "erp")
    for call in (lambda: scoped.dictionary(["ORDERS", "SECRET"]), lambda: scoped.table_sizes(["SECRET"]),
                 lambda: scoped.histograms([("ORDERS", "A"), ("SECRET", "B")])):
        with pytest.raises(TablesNotFound):
            call()
    assert inner.asked == [], "nothing was read once a requested table was out of scope"
    scoped.dictionary(["orders", "DIM_DATE"])
    scoped.table_sizes(["Orders"])
    scoped.histograms([("dim_region", "CODE")])
    assert [kind for kind, _ in inner.asked] == ["dictionary", "table_sizes", "histograms"]


def test_the_table_scope_offers_a_batched_read_only_where_the_source_has_one():
    class Plain:
        def table_size(self, table):
            return None

    scoped = TableScopedAdapter(Plain(), ("ORDERS",), "erp")
    assert not callable(getattr(scoped, "dictionary", None))
    assert not callable(getattr(scoped, "table_sizes", None))
