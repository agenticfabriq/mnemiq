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


LOOKUP = """
CREATE TABLE status (id INTEGER PRIMARY KEY, label TEXT);
CREATE TABLE tickets (id INTEGER PRIMARY KEY, status_id INTEGER REFERENCES status(id));
INSERT INTO status VALUES (1, 'Open'), (2, 'Closed');
INSERT INTO tickets VALUES (1, 1), (2, 2), (3, 1), (4, 1);
"""


def test_codes_are_grounded_from_a_lookup_table_enriched_in_another_subset(shop):
    # Codex's fourth review: lookup grounding reads a code's meaning across a join, and no single run saw this one.
    path, settings, _runs = shop
    con = sqlite3.connect(path)
    con.executescript(LOOKUP)
    con.commit()
    con.close()
    assert _cmd_enrich(settings(["status"])) == 0
    assert _cmd_enrich(settings(["tickets"]), add=True) == 0
    column = next(c for c in _latest(settings(["tickets"])).columns if c.id == "tickets.status_id")
    assert {(cv.code, cv.meaning) for cv in column.coded_values} >= {("1", "Open"), ("2", "Closed")}


# Codex's fifth review: what --add gives a column must be what one run over the same tables gives it. Each test runs
# both, a single run in its own store as the control, and compares the column.
def _coded(settings: Settings, column_id: str) -> set[tuple]:
    column = next(c for c in _latest(settings).columns if c.id == column_id)
    return {(cv.code, cv.meaning, cv.source) for cv in column.coded_values}


def _one_run_and_add(shop, script: str, first: list[str], second: list[str], column_id: str):
    path, settings, _runs = shop
    con = sqlite3.connect(path)
    con.executescript(script)
    con.commit()
    con.close()
    union = settings([*first, *second])
    union = union.model_copy(update={"store_path": union.store_path.replace("store.duckdb", "single.duckdb")})
    assert _cmd_enrich(union) == 0
    assert _cmd_enrich(settings(first)) == 0
    assert _cmd_enrich(settings(second), add=True) == 0
    return _coded(union, column_id), _coded(settings(second), column_id)


def test_a_key_column_pointing_into_another_subset_keeps_no_harvested_identifiers(shop):
    # Profiling leaves a declared key's values unharvested, but a run's scoped view drops a key into another subset.
    single, added = _one_run_and_add(shop, """
        CREATE TABLE person (code TEXT PRIMARY KEY);
        CREATE TABLE events (id INTEGER PRIMARY KEY, owner TEXT REFERENCES person(code));
        INSERT INTO person VALUES ('person-123'), ('person-456');
        INSERT INTO events VALUES (1, 'person-123'), (2, 'person-456'), (3, 'person-123');
    """, ["person"], ["events"], "events.owner")
    assert added == single


def test_grounding_across_subsets_matches_one_run_with_an_unmatched_lookup_code(shop):
    # Grounding twice is not grounding once: lookup grounding creates codes, which a second pass's correlated grounding
    # then reads. 99 is in no lookup row; the sibling label column says what it is. The lookup pair is enriched in one
    # run, and an unrelated table added after must leave its column as it was.
    single, added = _one_run_and_add(shop, """
        CREATE TABLE stage (id INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE cases (id INTEGER PRIMARY KEY, stage_id INTEGER REFERENCES stage(id), stage_label TEXT);
        INSERT INTO stage VALUES (1, 'Open');
        INSERT INTO cases VALUES (1, 1, 'Local open'), (2, 99, 'Escalated'), (3, 1, 'Local open'), (4, 99, 'Escalated');
    """, ["stage", "cases"], ["customers"], "cases.stage_id")
    assert added == single


def _normalised(snap) -> dict:
    """A snapshot as a single run and --add must agree on it: everything but what is per run by design (descriptions,
    which the stub tags with the run; the version and timestamp; job order)."""
    return {
        "bindings": sorted(b.object_id for b in snap.source_bindings),
        "columns": sorted((c.model_copy(update={"description": None}).model_dump_json() for c in snap.columns)),
        "relationships": sorted(r.model_dump_json() for r in snap.relationships),
        "views": sorted(v.model_dump_json() for v in snap.views),
        "dimensions": sorted(d.model_dump_json() for d in snap.dimensions),
        "definitions": sorted(d.model_dump_json() for d in snap.definitions),
        "jobs": sorted((j.id, j.status) for j in snap.jobs if not j.id.startswith("semantic:")),
    }


@pytest.mark.parametrize("first,second", [
    (["customers", "orders", "stage", "cases", "person"],
     ["product", "order_lines", "status", "tickets", "claim*", "events"]),
    (["status", "orders", "events"], ["tickets", "customers", "claim*", "product", "order_lines", "person", "stage",
                                      "cases"]),
])
def test_add_over_subsets_gives_what_one_run_over_all_of_them_gives(shop, first, second, tmp_path):
    path, base_settings, _runs = shop
    # The operator's dictionary restates a lookup code on a key column and a harvested code, so the overlays are
    # compared too.
    dictionary = tmp_path / "dictionary.json"
    dictionary.write_text(json.dumps({"columns": {"tickets.status_id": {"codes": {"1": "Opened"}},
                                                  "orders.status": {"codes": {"A": "Active (dictionary)"}}}}))

    def settings(tables):
        return base_settings(tables).model_copy(update={"dictionary_path": str(dictionary)})

    con = sqlite3.connect(path)
    con.executescript(LOOKUP + """
        CREATE TABLE claim (id INTEGER PRIMARY KEY, region TEXT, amount REAL);
        CREATE VIEW claim_v AS SELECT id, region, amount FROM claim;
        INSERT INTO claim VALUES (1, 'west', 10.0), (2, 'east', 20.0);
        CREATE TABLE person (code TEXT PRIMARY KEY);
        CREATE TABLE events (id INTEGER PRIMARY KEY, owner TEXT REFERENCES person(code));
        INSERT INTO person VALUES ('person-123'), ('person-456');
        INSERT INTO events VALUES (1, 'person-123'), (2, 'person-456'), (3, 'person-123');
        CREATE TABLE stage (id INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE cases (id INTEGER PRIMARY KEY, stage_id INTEGER REFERENCES stage(id), stage_label TEXT);
        INSERT INTO stage VALUES (1, 'Open');
        INSERT INTO cases VALUES (1, 1, 'Local open'), (2, 99, 'Escalated'), (3, 1, 'Local open'), (4, 99, 'Escalated');
    """)
    con.commit()
    con.close()
    union = settings([*first, *second])
    union = union.model_copy(update={"store_path": union.store_path.replace("store.duckdb", "single.duckdb")})
    assert _cmd_enrich(union) == 0
    assert _cmd_enrich(settings(first)) == 0
    assert _cmd_enrich(settings(second), add=True) == 0
    single, added = _normalised(_latest(union)), _normalised(_latest(settings(second)))
    for part in single:
        assert added[part] == single[part], part


def test_a_dictionary_override_of_a_lookup_code_survives_an_unrelated_add(shop, tmp_path):
    # Codex's sixth review: a key column keeps the codes its lookup gives it, whatever overlay has since restated them;
    # the dictionary marks the code it restates `dictionary`, and taking every code not marked `lookup` lost it.
    path, settings, _runs = shop
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE states (code TEXT PRIMARY KEY, label TEXT);
        CREATE TABLE tasks (id INTEGER PRIMARY KEY, state TEXT REFERENCES states(code));
        INSERT INTO states VALUES ('A', 'Active'), ('B', 'Blocked');
        INSERT INTO tasks VALUES (1, 'A'), (2, 'B'), (3, 'A');
    """)
    con.commit()
    con.close()
    dictionary = tmp_path / "dictionary.json"
    dictionary.write_text(json.dumps({"columns": {"tasks.state": {"codes": {"A": "Approved"}}}}))

    def with_dictionary(s: Settings, store: str | None = None) -> Settings:
        update = {"dictionary_path": str(dictionary)}
        if store:
            update["store_path"] = s.store_path.replace("store.duckdb", store)
        return s.model_copy(update=update)

    union = with_dictionary(settings(["states", "tasks", "customers"]), "single.duckdb")
    assert _cmd_enrich(union) == 0
    assert ("A", "Approved", "dictionary") in _coded(union, "tasks.state"), "the control: one run keeps the override"
    assert _cmd_enrich(with_dictionary(settings(["states", "tasks"]))) == 0
    assert _cmd_enrich(with_dictionary(settings(["customers"])), add=True) == 0
    assert _coded(settings(["customers"]), "tasks.state") == _coded(union, "tasks.state")
