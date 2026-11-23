"""`mnemiq enrich --add` end to end: subsets enriched one run at a time add up, joins between them included.

The model is a stub that writes each run's own description, so a description kept from an earlier run is visible. The
source is SQLite through `SQLiteAdapter`, which reports declared keys, patched in under `adapter_for` so the real
`TableScopedAdapter` still scopes each run (a `sqlite` source otherwise resolves to `DuckDBAdapter`, which reports none).
"""
import json
import sqlite3

import pytest

from mnemiq.adapters.sqlite import SQLiteAdapter
from mnemiq.cli import _cmd_enrich
from mnemiq.config import Settings
from mnemiq.enrichment.pipeline import content_version
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot
from tests.test_statistics_enrich import SHOP


@pytest.fixture
def shop(tmp_path, monkeypatch):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(SHOP)
    con.commit()
    con.close()
    monkeypatch.setattr("mnemiq.adapters.resolve._adapter_for", lambda *a, **k: SQLiteAdapter(str(path)))
    runs: list[str] = []

    def described(snap, enricher, **kwargs):
        runs.append("called")
        tag = f"run{len(runs)}"
        return snap.model_copy(update={"columns": [c.model_copy(update={"description": f"{tag}:{c.object_id}"})
                                                   for c in snap.columns]})

    monkeypatch.setattr("mnemiq.enrichment.semantic.enrich_semantic", described)

    def settings(tables: list[str]) -> Settings:
        manifest = tmp_path / "sources.json"
        manifest.write_text(json.dumps([{"id": "shop", "kind": "sqlite", "target": str(path), "catalog": "shop",
                                         "schema": "main", "tables": tables}]))
        return Settings(sources_path=str(manifest), store_path=str(tmp_path / "store.duckdb"),
                        llm_base_url="http://127.0.0.1:9/v1", llm_api_key="x", llm_model="m",
                        enrich_examples=False, enrich_facts=False)

    return path, settings, runs


def _latest(settings: Settings):
    con = init_store(settings.store_path)
    try:
        return load_snapshot(con, current_version(con, "shop"))
    finally:
        con.close()


def _description(snap, table: str) -> set[str]:
    return {c.description for c in snap.columns if c.object_id == table}


def test_two_add_runs_keep_both_subsets_and_their_descriptions(shop):
    _path, settings, _runs = shop
    assert _cmd_enrich(settings(["orders"])) == 0
    assert _cmd_enrich(settings(["customers"]), add=True) == 0

    snap = _latest(settings(["customers"]))
    assert {b.object_id for b in snap.source_bindings} == {"orders", "customers"}
    assert _description(snap, "orders") == {"run1:orders"}, "the first run's descriptions were kept, not redone"
    assert _description(snap, "customers") == {"run2:customers"}
    assert ("orders", "customers") in [(r.from_, r.to) for r in snap.relationships], "the key across the subsets"


def test_add_refuses_when_the_source_has_no_snapshot(shop, capsys):
    _path, settings, runs = shop
    assert _cmd_enrich(settings(["orders"]), add=True) == 1
    assert "--add" in capsys.readouterr().err
    assert runs == [], "refused before any model call"


def test_add_refuses_before_enriching_when_a_kept_table_is_gone(shop, capsys):
    path, settings, runs = shop
    assert _cmd_enrich(settings(["orders"])) == 0
    con = sqlite3.connect(path)
    con.execute("DROP TABLE orders")
    con.commit()
    con.close()
    assert _cmd_enrich(settings(["customers"]), add=True) == 1
    assert "orders" in capsys.readouterr().err
    assert runs == ["called"], "the second run made no model call"


def test_without_add_a_run_still_replaces_the_snapshot(shop):
    _path, settings, _runs = shop
    assert _cmd_enrich(settings(["orders"])) == 0
    assert _cmd_enrich(settings(["customers"])) == 0
    assert {b.object_id for b in _latest(settings(["customers"])).source_bindings} == {"customers"}


def test_the_merged_snapshot_gets_a_new_version(shop):
    _path, settings, _runs = shop
    assert _cmd_enrich(settings(["orders"])) == 0
    first = _latest(settings(["orders"])).version
    assert _cmd_enrich(settings(["customers"]), add=True) == 0
    merged = _latest(settings(["customers"]))
    assert merged.version != first
    assert merged.version == content_version(merged)


def test_the_command_line_flag_reaches_enrich(monkeypatch):
    from mnemiq import cli

    seen = {}
    monkeypatch.setattr(cli, "_cmd_enrich", lambda settings, add=False: seen.setdefault("add", add) and 0)
    cli.main(["enrich", "--add"])
    cli_seen = seen.pop("add")
    cli.main(["enrich"])
    assert (cli_seen, seen["add"]) == (True, False)


def test_add_discovers_views_again_and_keeps_an_earlier_failure_visible(shop, monkeypatch):
    # The view floor reads `discover:views` (`inventory_for`): the --add run's own discovery covers only its subset, so
    # an earlier failure must survive it, and the job must be the union pass's, not the subset's.
    _path, settings, _runs = shop

    def unanswered(self):
        raise RuntimeError("the source did not answer")

    with monkeypatch.context() as m:
        m.setattr(SQLiteAdapter, "view_definitions", unanswered)
        assert _cmd_enrich(settings(["orders"])) == 0
    assert next(j for j in _latest(settings(["orders"])).jobs if j.id == "discover:views").status == "failed"

    assert _cmd_enrich(settings(["customers"]), add=True) == 0
    job = next(j for j in _latest(settings(["customers"])).jobs if j.id == "discover:views")
    assert job.status == "failed" and "without --add" in (job.detail or "")


# The certified set and the operator's dictionary overlay every column, but a run applies them to its own subset only.
# Under --add they are laid again over the merged snapshot (Codex's third review): otherwise a column kept from an
# earlier run stays unclassified after a certified dimension marks it personal, and its values are indexed; and the
# join pass, which reads joins again over the union, would drop a certified join no key or name can recover.
from mnemiq.contract import CertifiedRecord  # noqa: E402
from mnemiq.enrichment.certified import CertifiedSet  # noqa: E402


def _certified(object_type: str, object_id: str, payload: dict) -> CertifiedRecord:
    return CertifiedRecord.model_validate({
        "envelope": {"object_type": object_type, "object_id": object_id, "version": "v1", "source_system": "verity",
                     "provenance": {"status": "certified", "certifier": "reviewer@acme"}},
        "payload": payload})


def _indexed_columns(settings: Settings) -> set[str]:
    con = init_store(settings.store_path)
    try:
        return {row[0] for row in con.execute("SELECT DISTINCT object_id || '.' || column_name FROM value_index").fetchall()}
    finally:
        con.close()


def test_a_certification_made_after_a_kept_run_classifies_its_columns_and_keeps_certified_joins(shop, monkeypatch):
    _path, settings, _runs = shop
    assert _cmd_enrich(settings(["customers"])) == 0
    assert "customers.region" in _indexed_columns(settings(["customers"])), "the control: open, so indexed"

    certified = CertifiedSet([
        _certified("dimension", "region_dim", {"id": "region_dim", "label": "Region", "source": "customers",
                                               "expr": "region", "pii_level": "pii"}),
        _certified("relationship", "orders.status->customers", {
            "id": "orders.status->customers", "from": "orders", "to": "customers", "cardinality": "many_to_one",
            "join_keys": [{"left": "status", "right": "region"}]}),
    ], available=True)
    monkeypatch.setattr("mnemiq.enrichment.certified.fetch_certified_records", lambda settings: certified)
    assert _cmd_enrich(settings(["orders"]), add=True) == 0

    snap = _latest(settings(["orders"]))
    region = next(c for c in snap.columns if c.id == "customers.region")
    assert region.pii_levels() == frozenset({"pii"}), "the kept column is classified by the dimension now certified"
    assert "customers.region" not in _indexed_columns(settings(["orders"])), "and its values are no longer indexed"
    assert [d.id for d in snap.dimensions] == ["region_dim"], "laid once, not appended twice"
    assert [r.id for r in snap.relationships].count("orders.status->customers") == 1, "the certified join is kept"


def test_add_refuses_before_any_model_call_when_the_ontology_changed(shop, capsys):
    from mnemiq.store.snapshot_store import save_snapshot

    _path, settings, runs = shop
    assert _cmd_enrich(settings(["customers"])) == 0
    con = init_store(settings(["customers"]).store_path)
    try:
        save_snapshot(con, _latest(settings(["customers"])).model_copy(update={"ontology_version": "older",
                                                                               "version": "v-older"}))
    finally:
        con.close()
    assert _cmd_enrich(settings(["orders"]), add=True) == 1
    assert "ontology" in capsys.readouterr().err
    assert runs == ["called"], "refused before the second run's model calls"
