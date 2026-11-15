"""How much of each piece of certified meaning reaches a prompt, and the warning when some does not.

The writer's prompt keeps the first N characters of each definition, measure and dimension. A definition
longer than its limit lost its last sentences silently: nothing at boot, nothing per question, and the
model was told to "follow them exactly" about a text that stopped mid-rule.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import duckdb

from mnemiq.config import Settings
from mnemiq.contract import Definition, Snapshot, SourceBinding
from mnemiq.contract.semantic import Dimension, Metric
from mnemiq.contract.values import MeasureExpr
from mnemiq.generate.prompts import (
    DEFINITION_LIMIT,
    DIMENSION_LIMIT,
    MEASURE_LIMIT,
    NAME_LIMIT,
    cut_meaning,
    meaning_sections,
)


def _definition(text: str, term: str = "lot id") -> Definition:
    return Definition(id=f"d:{term}", term=term, domain="plant", definition=text)


def _metric(expr: str, label: str = "yield") -> Metric:
    return Metric(id="m:yield", label=label, status="certified", owner="o", grain="g",
                  measure=MeasureExpr(expr=expr, source="lots"), time_dimension="t")


def _dimension(expr: str) -> Dimension:
    return Dimension(id="dim:grade", label="grade band", source="lots", expr=expr)


def _packet(definitions=(), metrics=(), dimensions=()):
    return SimpleNamespace(definitions=list(definitions), metrics=list(metrics),
                           dimensions=list(dimensions), concepts=[])


def _words(n: int) -> str:
    """`n` characters of prose ending on a marker, so a cut is visible as the marker's absence."""
    body = ("word " * n)[: n - len(" END")].rstrip()
    return (body + " END").rjust(n, "w")


def test_a_definition_of_707_characters_reaches_the_prompt_whole():
    # The length that was cut in the field: its last sentence held two join keys.
    text = _words(707)
    rendered = "\n".join(meaning_sections(_packet([_definition(text)])))
    assert text in rendered


def test_a_definition_at_the_limit_is_whole_and_not_reported():
    text = _words(DEFINITION_LIMIT)
    assert text in "\n".join(meaning_sections(_packet([_definition(text)])))
    assert cut_meaning(_packet([_definition(text)])) == []


def test_a_definition_over_the_limit_is_cut_and_reported_by_term_and_length():
    text = _words(DEFINITION_LIMIT + 1)
    rendered = "\n".join(meaning_sections(_packet([_definition(text, term="ingot id")])))
    assert "END" not in rendered, "the renderer cuts at the limit"
    (sentence,) = cut_meaning(_packet([_definition(text, term="ingot id")]))
    assert "'ingot id'" in sentence
    assert f"{DEFINITION_LIMIT + 1:,}" in sentence and f"{DEFINITION_LIMIT:,}" in sentence


def test_the_report_counts_what_the_renderer_keeps_not_the_raw_length():
    # Whitespace collapses before the cut, so a text long only in its spaces renders whole.
    text = "a" + " " * (DEFINITION_LIMIT * 2) + "END"
    assert "a END" in "\n".join(meaning_sections(_packet([_definition(text)])))
    assert cut_meaning(_packet([_definition(text)])) == []


def test_a_term_over_its_limit_is_reported():
    term = "t" * (NAME_LIMIT + 1)
    (sentence,) = cut_meaning(_packet([_definition("short", term=term)]))
    assert f"{NAME_LIMIT + 1:,}" in sentence


def test_a_measure_expression_over_its_limit_is_reported():
    expr = "sum(x) + " * (MEASURE_LIMIT // 9 + 1) + "0"
    rendered = "\n".join(meaning_sections(_packet(metrics=[_metric(expr)])))
    assert expr not in rendered
    (sentence,) = cut_meaning(_packet(metrics=[_metric(expr)]))
    assert "'m:yield'" in sentence and f"{MEASURE_LIMIT:,}" in sentence


def test_a_dimension_expression_over_its_limit_is_reported():
    expr = "CASE " + "WHEN g = 'A' THEN 1 " * (DIMENSION_LIMIT // 20 + 1) + "END"
    rendered = "\n".join(meaning_sections(_packet(dimensions=[_dimension(expr)])))
    assert expr not in rendered
    (sentence,) = cut_meaning(_packet(dimensions=[_dimension(expr)]))
    assert "'dim:grade'" in sentence and f"{DIMENSION_LIMIT:,}" in sentence


def test_nothing_cut_reports_nothing():
    packet = _packet([_definition("A lot is a unit of production.")], [_metric("sum(x)")],
                     [_dimension("grade")])
    assert cut_meaning(packet) == []


def _boot(tmp_path, snapshot: Snapshot) -> Settings:
    """A store holding `snapshot`, a source and a policy, as `build_runtime` reads them."""
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    source = tmp_path / "src.duckdb"
    duckdb.connect(str(source)).execute("CREATE TABLE lots (lot_id TEXT)").close()
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    save_snapshot(con, snapshot)
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "duckdb", "target": str(source),
                                     "catalog": "src", "schema": "main"}]))
    authz = tmp_path / "authz.json"
    authz.write_text(json.dumps({"roles": {"analyst": ["lots"]}}))
    return Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m",
                    llm_window_check=False, sources_path=str(manifest), store_path=str(store),
                    authz_path=str(authz))


def _snapshot(definitions) -> Snapshot:
    return Snapshot(version="v1", source_id="only", created_at="2026-11-15T00:00:00Z",
                    source_bindings=[SourceBinding(id="sb:lots", source_id="only", object_id="lots",
                                                   source_object="lots", binding_type="table")],
                    definitions=definitions)


def test_build_runtime_warns_about_a_definition_the_prompt_cuts(tmp_path, caplog):
    """The join: the boot door over a stored snapshot, nothing stubbed but the model server."""
    from mnemiq.runtime import build_runtime

    settings = _boot(tmp_path, _snapshot([_definition(_words(DEFINITION_LIMIT + 50), term="ingot id")]))
    with caplog.at_level(logging.WARNING):
        build_runtime(settings)
    cut = [r for r in caplog.records if r.levelno == logging.WARNING and "'ingot id'" in r.getMessage()]
    assert len(cut) == 1, caplog.text


def test_build_runtime_is_silent_when_nothing_is_cut(tmp_path, caplog):
    from mnemiq.runtime import build_runtime

    settings = _boot(tmp_path, _snapshot([_definition("A lot is a unit of production.", term="lot")]))
    with caplog.at_level(logging.WARNING):
        build_runtime(settings)
    assert "the prompt shows the first" not in caplog.text


def test_a_hot_swapped_snapshot_is_checked_again(monkeypatch, caplog):
    """`reload_if_stale` replaces the snapshot, definitions included: the new one is checked."""
    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    new = _snapshot([_definition(_words(DEFINITION_LIMIT + 50), term="swapped in")])
    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con, rt.authz = _Settings(), object(), None
    rt.snapshot, rt.loaded_versions = object(), {"only": "v1"}
    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot", lambda _s, _c: (new, {"only": "v2"}))
    monkeypatch.setattr("mnemiq.runtime._warn_prompt_window", lambda *a, **k: None)
    monkeypatch.setattr("mnemiq.runtime._warn_policy_advisories", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING):
        rt.reload_if_stale()
    assert "'swapped in'" in caplog.text
