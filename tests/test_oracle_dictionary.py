"""What the Oracle dictionary says about a table's size, read with no table data: `OracleAdapter.table_size`.

The dictionary is stubbed (`_rows` answered by SQL shape, as `test_oracle_resolve_called_functions.py` does); the live
read against Oracle is `test_oracle_adapter.py::test_table_size_live`.
"""
from __future__ import annotations

import datetime

import pytest

from mnemiq.adapters.dictionary import TableSize
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
    assert _adapter(stats).table_size("ORDERS") == TableSize(rows=1000, avg_row_len=80, analyzed="2026-11-01",
                                                             stale=False)
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


def test_scoped_adapter_reads_a_table_in_scope():
    inner = _Inner()
    assert TableScopedAdapter(inner, ("orders",), "src").table_size("ORDERS").rows == 10
    assert inner.asked == ["ORDERS"]


def test_scoped_adapter_refuses_a_table_out_of_scope():
    inner = _Inner()
    with pytest.raises(TablesNotFound, match="PAYROLL"):
        TableScopedAdapter(inner, ("orders",), "src").table_size("PAYROLL")
    assert inner.asked == []
