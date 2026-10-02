"""The boot advisory that says when the largest generation prompt cannot fit the window (M119)."""

from __future__ import annotations

import json
import logging
import math

import duckdb
import httpx
import pytest

from mnemiq.config import Settings
from mnemiq.contract import Column, Definition, Snapshot, SourceBinding
from mnemiq.generate.generator import GENERATOR_MAX_TOKENS
from mnemiq.llm import window
from mnemiq.llm.window import WindowReport, check_window, count_on_server, largest_prompt


def _snapshot() -> Snapshot:
    """Three tables of very different widths, a definition bound to each end, one unbound."""
    widths = {"wide": 40, "middle": 12, "narrow": 2}
    return Snapshot(
        version="v1", source_id="s", created_at="2026-10-02T00:00:00Z",
        source_bindings=[
            SourceBinding(id=f"sb:{t}", source_id="s", object_id=t, source_object=t,
                          binding_type="table")
            for t in widths
        ],
        columns=[
            Column(id=f"{t}.col_{i}", object_id=t, name=f"col_{i}", data_type="text",
                   description=f"Column {i} of {t}, described at some length.")
            for t, n in widths.items() for i in range(n)
        ],
        definitions=[
            Definition(id="d:wide", term="wide policy", domain="ops", definition="WIDE-DEFINITION-TEXT",
                       bound_objects=["wide"]),
            Definition(id="d:narrow", term="narrow policy", domain="ops",
                       definition="NARROW-DEFINITION-TEXT",
                       bound_objects=["narrow"]),
        ],
    )


def _settings(**over) -> Settings:
    return Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m",
                    retrieval_k=2, **over)


def _con_with_examples(examples: list[tuple[str, str]]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE example (question TEXT, sql TEXT, tables TEXT, object_id TEXT, "
                "source_id TEXT)")
    for q, s in examples:
        con.execute("INSERT INTO example VALUES (?, ?, ?, ?, ?)", [q, s, json.dumps(["wide"]),
                                                                   "wide", "s"])
    return con


def test_the_largest_prompt_holds_the_k_widest_cards_and_what_rides_with_them():
    con = _con_with_examples([("short?", "SELECT 1"), ("a much longer question?" * 5, "SELECT " + "x, " * 50),
                              ("medium question?", "SELECT a, b")])
    system, user, cards = largest_prompt(con, _snapshot(), _settings(), "duckdb")

    assert cards == 2
    assert "wide" in user and "middle" in user
    assert "narrow" not in user.split("TABLES:")[1], "the narrowest card is past k=2"
    assert "WIDE-DEFINITION-TEXT" in user, "a definition bound to a shown table rides with it"
    assert "NARROW-DEFINITION-TEXT" not in user, "its table was not shown"
    assert "a much longer question?" in user and "medium question?" in user
    assert "short?" not in user, "only the k largest examples"
    assert "x" * 500 in user, "the question at the length sanitize keeps"
    assert system


def test_a_store_without_examples_still_measures():
    _, user, cards = largest_prompt(duckdb.connect(), _snapshot(), _settings(), "duckdb")
    assert cards == 2 and "WORKED EXAMPLES" not in user


def _server(reply: dict | None, status: int = 200, seen: list | None = None) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(status, json=reply or {})
    return httpx.Client(transport=httpx.MockTransport(handle))


def test_vllm_counts_at_its_root_tokenize_with_the_chat_messages():
    seen: list = []
    got = count_on_server("http://h:8002/v1", "m", "SYS", "USER",
                          http=_server({"count": 21813, "max_model_len": 28672}, seen=seen))
    assert got == (21813, 28672)
    path, body = seen[0]
    assert path == "/tokenize", "beside /v1, not under it"
    assert body["messages"] == [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "USER"}]


def test_a_server_that_is_down_or_not_json_is_no_count():
    def refused(request):
        raise httpx.ConnectError("refused")

    down = httpx.Client(transport=httpx.MockTransport(refused))
    html = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="<html>catch-all</html>")))
    listed = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[1])))
    for client in (down, html, listed):
        assert count_on_server("http://h/v1", "m", "s", "u", http=client) is None


def test_a_declared_window_is_still_checked_when_the_server_is_down():
    def refused(request):
        raise httpx.ConnectError("refused")

    report = check_window(duckdb.connect(), _snapshot(), _settings(llm_context_window=100),
                          "duckdb", http=httpx.Client(transport=httpx.MockTransport(refused)))
    assert report.window == 100 and report.fits is False


def test_a_prompt_and_reply_exactly_filling_the_window_fit():
    assert WindowReport(24_672, 4_000, 28_672, counted_by_server=True, cards=5).fits is True
    assert WindowReport(24_673, 4_000, 28_672, counted_by_server=True, cards=5).fits is False


@pytest.mark.parametrize(("reply", "status"), [
    (None, 404),                        # no such door: Ollama, a hosted API
    ({"count": 10}, 200),               # a count with no window says nothing to compare
    ({"count": "10", "max_model_len": 5}, 200),
])
def test_a_server_that_cannot_count_and_report_is_not_read_as_one(reply, status):
    assert count_on_server("http://h/v1", "m", "s", "u", http=_server(reply, status)) is None


def test_the_server_count_and_window_decide_when_the_server_gives_them():
    con = duckdb.connect()
    report = check_window(con, _snapshot(), _settings(llm_context_window=999_999), "duckdb",
                          http=_server({"count": 25_000, "max_model_len": 28_672}))
    assert report.counted_by_server and report.window == 28_672, "the server outranks a declaration"
    assert report.reply_tokens == GENERATOR_MAX_TOKENS
    assert report.fits is False  # 25,000 + 4,000 > 28,672


def test_without_a_server_count_the_estimate_errs_long_against_the_declared_window():
    con = duckdb.connect()
    snap, settings = _snapshot(), _settings(llm_context_window=100)
    report = check_window(con, snap, settings, "duckdb", http=_server(None, 404))
    system, user, _ = largest_prompt(con, snap, settings, "duckdb")
    assert not report.counted_by_server and report.window == 100
    assert report.prompt_tokens == math.ceil((len(system) + len(user)) / window.CHARS_PER_TOKEN_FLOOR)
    assert report.fits is False


def test_no_count_and_no_declaration_is_not_a_verdict():
    report = check_window(duckdb.connect(), _snapshot(), _settings(), "duckdb",
                          http=_server(None, 404))
    assert report.window is None and report.fits is None


# --- the advisory: how a report reaches the operator --------------------------------------------

def _report(fits: bool, by_server: bool = True) -> WindowReport:
    prompt = 21_000 if fits else 30_000
    return WindowReport(prompt, 4_000, 28_672, counted_by_server=by_server, cards=5)


@pytest.mark.parametrize(("by_server", "label"), [(True, "TOO SMALL"), (False, "MAY BE TOO SMALL")])
def test_a_prompt_past_the_window_warns_and_says_how_sure(caplog, monkeypatch, by_server, label):
    from mnemiq.runtime import _warn_prompt_window

    monkeypatch.setattr(window, "check_window", lambda *a, **k: _report(False, by_server))
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), None, _snapshot(), None, frozenset())
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and f"prompt window {label}:" in warned[0].getMessage()
    assert "MNEMIQ_RETRIEVAL_K" in warned[0].getMessage()


def test_an_acknowledged_small_window_drops_to_info(caplog, monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    monkeypatch.setattr(window, "check_window", lambda *a, **k: _report(False))
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), None, _snapshot(), None, frozenset({"window:too-small"}))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "prompt window TOO SMALL" in caplog.text


def test_a_prompt_that_fits_does_not_warn(caplog, monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    monkeypatch.setattr(window, "check_window", lambda *a, **k: _report(True))
    with caplog.at_level(logging.INFO):
        _warn_prompt_window(_settings(), None, _snapshot(), None, frozenset())
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "prompt window fits" in caplog.text


def test_the_advisory_never_raises(monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(window, "check_window", boom)
    _warn_prompt_window(_settings(), None, _snapshot(), None, frozenset())  # no exception


def test_switched_off_it_asks_nothing(monkeypatch):
    from mnemiq.runtime import _warn_prompt_window

    def must_not_run(*a, **k):
        raise AssertionError("checked while switched off")

    monkeypatch.setattr(window, "check_window", must_not_run)
    _warn_prompt_window(_settings(llm_window_check=False), None, _snapshot(), None, frozenset())


def test_build_runtime_runs_it_over_the_store_it_built(caplog, monkeypatch, tmp_path):
    """The join: the boot door, with only the server's reply stubbed. The prompt measured is the
    real one from the store `build_runtime` opened, and the warning reaches the log."""
    from mnemiq.runtime import build_runtime
    from mnemiq.store.bootstrap import init_store
    from mnemiq.store.snapshot_store import save_snapshot

    source = tmp_path / "src.duckdb"
    duckdb.connect(str(source)).execute("CREATE TABLE wide (col_0 TEXT)").close()
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    save_snapshot(con, _snapshot().model_copy(update={"source_id": "only"}))
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "duckdb", "target": str(source),
                                     "catalog": "src", "schema": "main"}]))

    measured: list[str] = []

    def server(base_url, model, system, user, http=None, timeout=10.0):
        measured.append(user)
        return (30_000, 28_672)

    monkeypatch.setattr(window, "count_on_server", server)
    settings = _settings(sources_path=str(manifest), store_path=str(store))
    with caplog.at_level(logging.WARNING):
        build_runtime(settings)

    assert measured and "WIDE-DEFINITION-TEXT" in measured[0], "measured over the loaded snapshot"
    assert "prompt window TOO SMALL" in caplog.text


def test_a_hot_swapped_snapshot_re_measures(monkeypatch):
    """The cards are the subject, and `reload_if_stale` replaces them: a store rebuilt with wider
    tables must be measured again, not judged by the boot snapshot."""
    from mnemiq.runtime import Runtime

    class _Settings:
        control_dsn = "postgresql://x"
        ack_advisories = ""

    new_snapshot = object()
    seen = []
    rt = Runtime.__new__(Runtime)
    rt.settings, rt.con, rt.authz = _Settings(), object(), None
    rt.snapshot, rt.loaded_versions = object(), {"src": "v1"}
    monkeypatch.setattr("mnemiq.runtime.load_current_snapshot",
                        lambda _s, _c: (new_snapshot, {"src": "v2"}))
    monkeypatch.setattr("mnemiq.runtime._warn_prompt_window",
                        lambda settings, con, snapshot, adapter, ack: seen.append(snapshot))

    rt.reload_if_stale()

    assert seen == [new_snapshot]
