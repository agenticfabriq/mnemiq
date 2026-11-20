"""Before an exact profile of an Oracle source, enrich reads each table's size from Oracle's statistics and stops
before reading any table if one is large (`mnemiq.enrichment.size_check`).

Exact profiling reads every table in full many times; on a production database's large tables that was measured at
hours for one read. So until the manifest says how to profile an Oracle source, a large table stops enrich, saying how
to go on. Sources of every other kind, and an Oracle source whose manifest names a "profile", read nothing extra.
"""
from __future__ import annotations

import json

import pytest

from mnemiq.adapters.dictionary import TableSize
from mnemiq.config import SourceSpec
from mnemiq.enrichment.size_check import LARGE_TABLE_BYTES, exact_profile_refusal

GiB = 1 << 30


def _spec(kind: str = "oracle", profile: str = "") -> SourceSpec:
    return SourceSpec(id="erp", kind=kind, target="db", catalog="erp", schema="APP", profile=profile)


class _Dictionary:
    """An Oracle source as the size check sees it: its tables, and each one's size. Reading table data fails."""

    dialect = "oracle"

    def __init__(self, sizes: dict[str, TableSize]):
        self.sizes = sizes
        self.asked: list[str] = []
        self.listed = 0

    def introspect(self):
        self.listed += 1
        return list(self.sizes)

    def table_size(self, table):
        self.asked.append(table)
        return self.sizes[table]

    def execute(self, sql):
        raise AssertionError(f"the size check read table data: {sql}")


def _size(rows, avg_row_len=100):
    return TableSize(rows=rows, avg_row_len=avg_row_len, analyzed="2026-11-01", stale=False)


def test_the_line_is_one_gib():
    assert LARGE_TABLE_BYTES == GiB


def test_a_large_table_stops_enrich_before_any_table_is_read():
    source = _Dictionary({"ORDERS": _size(20_000_000), "CUSTOMERS": _size(1_000, 50)})
    refusal, warning = exact_profile_refusal(_spec(), source)
    assert refusal.startswith("enrich stopped before reading any table: 1 table(s) of source 'erp'")
    assert "ORDERS (~1.9 GiB)" in refusal and "CUSTOMERS" not in refusal
    assert '"profile": "exact"' in refusal and '"tables"' in refusal
    assert '"profile": "statistics"' in refusal and '"exact_tables"' in refusal
    assert warning is None


def test_a_table_at_the_line_is_not_large():
    source = _Dictionary({"ORDERS": TableSize(rows=GiB // 64, avg_row_len=64, analyzed="2026-11-01", stale=False)})
    assert exact_profile_refusal(_spec(), source) == (None, None)


@pytest.mark.parametrize("profile", ["exact", "statistics"])
def test_a_profile_named_in_the_manifest_skips_the_check(profile):
    source = _Dictionary({"ORDERS": _size(20_000_000)})
    assert exact_profile_refusal(_spec(profile=profile), source) == (None, None)
    assert (source.listed, source.asked) == (0, [])


@pytest.mark.parametrize("kind", ["duckdb", "sqlite", "postgres"])
def test_off_oracle_the_check_reads_nothing(kind):
    source = _Dictionary({"ORDERS": _size(20_000_000)})
    assert exact_profile_refusal(_spec(kind=kind), source) == (None, None)
    assert (source.listed, source.asked) == (0, [])


def test_tables_without_statistics_warn_and_go_on():
    source = _Dictionary({"ORDERS": TableSize(None, None, None, None), "CUSTOMERS": _size(1_000)})
    refusal, warning = exact_profile_refusal(_spec(), source)
    assert refusal is None
    assert "1 table(s) of source 'erp' have no table-level optimizer statistics" in warning and "ORDERS" in warning
    assert "DBMS_STATS.GATHER_TABLE_STATS" in warning and '"tables"' in warning


def test_stale_tables_are_named_in_the_warning():
    """A table measured small and since grown passes the line on its old size; Oracle's stale flag is the only sign.
    It is named, with the size its statistics give, and what to do."""
    source = _Dictionary({"ORDERS": TableSize(rows=4_000_000, avg_row_len=100, analyzed="2026-01-01", stale=True),
                          "CUSTOMERS": _size(1_000)})
    refusal, warning = exact_profile_refusal(_spec(), source)
    assert refusal is None
    assert "1 table(s) of source 'erp' have statistics Oracle marks stale" in warning
    assert "ORDERS (~0.4 GiB as of 2026-01-01)" in warning and "CUSTOMERS" not in warning
    assert "DBMS_STATS.GATHER_TABLE_STATS" in warning


def test_unknown_staleness_is_not_warned_as_stale():
    source = _Dictionary({"ORDERS": TableSize(rows=1_000, avg_row_len=100, analyzed="2026-01-01", stale=None)})
    assert exact_profile_refusal(_spec(), source) == (None, None)


def test_size_check_reads_each_table_once():
    sizes = {f"T{i:03d}": _size(1_000) for i in range(600)}
    sizes["T599"] = _size(30_000_000)
    source = _Dictionary(sizes)
    refusal, _ = exact_profile_refusal(_spec(), source)
    assert sorted(source.asked) == sorted(sizes) and len(source.asked) == 600
    assert refusal.count("T599") == 1


# --- the wiring: `mnemiq enrich` -----------------------------------------------------------------

def _settings(tmp_path, **entry):
    from mnemiq.config import Settings

    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "erp", "kind": "oracle", "target": "db:1521/APP", "catalog": "erp",
                                     "schema": "APP", **entry}]))
    return Settings(sources_path=str(manifest), store_path=str(tmp_path / "store.duckdb"), llm_base_url=None,
                    llm_api_key=None, llm_model=None, acme_data_dir=None, pg_dsn=None)


class _Profiled(Exception):
    """Raised by the stubbed profile: reaching it is the fact these tests read, and nothing after it matters."""


def _wire(monkeypatch, source):
    profiled: list = []

    def profile(*a, **k):
        profiled.append((a, k))
        raise _Profiled

    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", lambda *a, **k: source)
    monkeypatch.setattr("mnemiq.enrichment.pipeline.enrich_structural", profile)
    return profiled


def test_enrich_exits_1_and_reads_no_table_on_refusal(monkeypatch, tmp_path, capsys):
    from mnemiq.cli import _cmd_enrich

    profiled = _wire(monkeypatch, _Dictionary({"ORDERS": _size(20_000_000)}))
    assert _cmd_enrich(_settings(tmp_path)) == 1
    assert profiled == []
    assert "enrich stopped before reading any table" in capsys.readouterr().err


def test_enrich_passes_the_statistics_scope(monkeypatch, tmp_path):
    """"statistics" skips the size check, which exists for exact profiling, and reaches the profile with the tables the
    manifest names exact."""
    from mnemiq.cli import _cmd_enrich

    source = _Dictionary({"ORDERS": _size(20_000_000)})
    profiled = _wire(monkeypatch, source)
    with pytest.raises(_Profiled):
        _cmd_enrich(_settings(tmp_path, profile="statistics", exact_tables=["orders", "dim_*"]))
    assert profiled[0][1]["statistics"] == ("orders", "dim_*") and source.asked == []
    profiled.clear()
    with pytest.raises(_Profiled):
        _cmd_enrich(_settings(tmp_path, profile="statistics"))
    assert profiled[0][1]["statistics"] == ()


@pytest.mark.parametrize("entry", [{}, {"profile": "exact"}])
def test_enrich_profiles_exactly_unless_statistics_is_named(monkeypatch, tmp_path, entry):
    from mnemiq.cli import _cmd_enrich

    profiled = _wire(monkeypatch, _Dictionary({"ORDERS": _size(1_000)}))
    with pytest.raises(_Profiled):
        _cmd_enrich(_settings(tmp_path, **entry))
    assert profiled[0][1].get("statistics") is None


def test_enrich_prints_the_warning_and_goes_on(monkeypatch, tmp_path, capsys):
    from mnemiq.cli import _cmd_enrich

    profiled = _wire(monkeypatch, _Dictionary({"ORDERS": TableSize(None, None, None, None)}))
    with pytest.raises(_Profiled):
        _cmd_enrich(_settings(tmp_path))
    assert len(profiled) == 1
    assert "WARNING: 1 table(s) of source 'erp' have no table-level optimizer statistics" in capsys.readouterr().err


# --- after the profile: how each table was profiled, and which lack statistics ---------------------

def _profiled_snapshot():
    """Two tables profiled exactly, three from statistics: one current, one with none, one stale."""
    from mnemiq.contract import Column, Job, Snapshot
    from mnemiq.contract.semantic import ColumnEstimate

    def est(rows, stale=False, analyzed="2026-11-01"):
        return ColumnEstimate(rows=rows, distinct=2, nulls=0, analyzed=analyzed, sampled=False, stale=stale)

    columns = [
        Column(id="ORDERS.ID", object_id="ORDERS", name="ID", row_count=10, distinct_count=10, null_count=0),
        Column(id="CUSTOMERS.ID", object_id="CUSTOMERS", name="ID", row_count=5, distinct_count=5, null_count=0),
        Column(id="ORDER_LINES.ID", object_id="ORDER_LINES", name="ID", estimate=est(1000)),
        Column(id="PAYMENTS.ID", object_id="PAYMENTS", name="ID"),
        Column(id="RETURNS.ID", object_id="RETURNS", name="ID", estimate=est(400, stale=True, analyzed="2026-01-01")),
    ]
    jobs = [Job(id=f"profile:{t}", source_id="erp", kind="profile", status="done", detail=d)
            for t, d in (("ORDERS", None), ("CUSTOMERS", None), ("ORDER_LINES", "estimated"),
                         ("PAYMENTS", "estimated"), ("RETURNS", "estimated"))]
    return Snapshot(version="v", source_id="erp", created_at="t", columns=columns, jobs=jobs)


def test_the_report_counts_each_way_and_names_what_lacks_statistics():
    from mnemiq.enrichment.statistics import profile_report

    summary, warning = profile_report(_profiled_snapshot())
    assert summary == "profiled 2 table(s) exactly, 3 from Oracle's statistics"
    assert "1 table(s) of source 'erp' have no optimizer statistics" in warning and "PAYMENTS" in warning
    assert "1 table(s) of source 'erp' have statistics Oracle marks stale" in warning
    assert "RETURNS (as of 2026-01-01)" in warning and "ORDER_LINES" not in warning
    assert "DBMS_STATS.GATHER_TABLE_STATS" in warning and '"exact_tables"' in warning


def test_no_report_when_every_table_was_profiled_exactly():
    from mnemiq.contract import Job, Snapshot
    from mnemiq.enrichment.statistics import profile_report

    snap = Snapshot(version="v", source_id="erp", created_at="t",
                    jobs=[Job(id="profile:ORDERS", source_id="erp", kind="profile", status="done")])
    assert profile_report(snap) == (None, None)


def _enrich_reporting(monkeypatch, tmp_path, snapshot):
    from mnemiq.cli import _cmd_enrich

    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", lambda *a, **k: _Dictionary({}))
    monkeypatch.setattr("mnemiq.enrichment.pipeline.enrich_structural", lambda *a, **k: snapshot)

    def stop(*a, **k):
        raise _Profiled

    monkeypatch.setattr("mnemiq.enrichment.certified.fetch_certified_records", stop)
    with pytest.raises(_Profiled):
        _cmd_enrich(_settings(tmp_path, profile="statistics"))


def test_enrich_reports_how_each_table_was_profiled(monkeypatch, tmp_path, capsys):
    _enrich_reporting(monkeypatch, tmp_path, _profiled_snapshot())
    assert "profiled 2 table(s) exactly, 3 from Oracle's statistics" in capsys.readouterr().out


def test_enrich_names_tables_without_statistics(monkeypatch, tmp_path, capsys):
    _enrich_reporting(monkeypatch, tmp_path, _profiled_snapshot())
    err = capsys.readouterr().err
    assert "WARNING: 1 table(s) of source 'erp' have no optimizer statistics" in err and "PAYMENTS" in err
    assert "Gather their statistics" in err
