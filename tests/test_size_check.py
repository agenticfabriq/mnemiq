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
    # statistics mode is not in this version: the message must not send anyone to it
    assert "statistics\"" not in refusal
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
    assert "1 table(s) of source 'erp' have no optimizer statistics" in warning and "ORDERS" in warning


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
        profiled.append(a)
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


def test_enrich_refuses_statistics_until_it_is_built(monkeypatch, tmp_path, capsys):
    """Taking "statistics" as "skip the check" would profile every table in full: the harm the check exists for."""
    from mnemiq.cli import _cmd_enrich

    source = _Dictionary({"ORDERS": _size(20_000_000)})
    profiled = _wire(monkeypatch, source)
    assert _cmd_enrich(_settings(tmp_path, profile="statistics")) == 1
    assert profiled == [] and source.asked == []
    assert '"profile": "statistics" is not available in this version' in capsys.readouterr().err


def test_enrich_prints_the_warning_and_goes_on(monkeypatch, tmp_path, capsys):
    from mnemiq.cli import _cmd_enrich

    profiled = _wire(monkeypatch, _Dictionary({"ORDERS": TableSize(None, None, None, None)}))
    with pytest.raises(_Profiled):
        _cmd_enrich(_settings(tmp_path))
    assert len(profiled) == 1
    assert "WARNING: 1 table(s) of source 'erp' have no optimizer statistics" in capsys.readouterr().err
