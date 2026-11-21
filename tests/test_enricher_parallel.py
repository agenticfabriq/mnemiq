"""The describer runs the chunk calls of many tables at once (`enrich_concurrency`).

Enrich described one table after another, and each table in chunks of `CHUNK_COLUMNS` one after another: on a schema
of 600 tables that is thousands of calls in a row, and one wide table is dozens alone. `annotate_tables` runs every
table's chunks on one pool of `concurrency` workers, joins each table's results in chunk order, and keeps `annotate`'s
rule that a failing endpoint stops the rest of that table. At 1 it asks exactly what `annotate` asks, in order.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
import time

import pytest

from mnemiq.enrichment.enricher import CHUNK_COLUMNS, LLMEnricher
from mnemiq.enrichment.prompts import ColumnFacts

_ASKED = re.compile(r"^- (\S+) \(type=", re.MULTILINE)


class _Model:
    """Documents exactly the columns a prompt asks about, as a well-behaved model would; thread-safe, so it can be asked
    from several workers. `delay` makes each call take that long; `fail_after` makes every call after that many raise."""

    def __init__(self, delay: float = 0.0, fail_after: int | None = None):
        self.prompts: list[str] = []
        self._delay, self._fail_after = delay, fail_after
        self._lock = threading.Lock()

    def complete(self, system, user, max_tokens=512, **_):
        with self._lock:
            self.prompts.append(user)
            n = len(self.prompts)
        if self._delay:
            time.sleep(self._delay)
        if self._fail_after is not None and n > self._fail_after:
            raise RuntimeError("the endpoint is down")
        names = _ASKED.findall(user.split("The table's other columns")[0])
        return json.dumps({"columns": [{"name": n, "description": f"about {n}"} for n in names]})


def _facts(prefix: str, n: int) -> list[ColumnFacts]:
    return [ColumnFacts(name=f"{prefix}{i}", data_type="text") for i in range(n)]


_REQUESTS = [("wide", _facts("w", 70), ""), ("small", _facts("s", 10), ""), ("mid", _facts("m", 40), "")]


def _described(annotation) -> list[tuple[str, str]]:
    return [(c.name, c.description) for c in annotation.columns]


def test_concurrency_one_asks_what_annotate_asks():
    one, each = _Model(), _Model()
    LLMEnricher(one).annotate_tables(_REQUESTS, concurrency=1)
    for table, facts, grounding in _REQUESTS:
        LLMEnricher(each).annotate(table, facts, grounding)
    assert one.prompts == each.prompts


def test_parallel_annotations_are_the_sequential_ones_in_table_order():
    sequential = LLMEnricher(_Model()).annotate_tables(_REQUESTS, concurrency=1)
    parallel = LLMEnricher(_Model()).annotate_tables(_REQUESTS, concurrency=4)
    assert [a.table for a in parallel] == ["wide", "small", "mid"]
    assert [_described(a) for a in parallel] == [_described(a) for a in sequential]
    assert len(parallel[0].columns) == 70


def test_chunks_of_one_wide_table_run_side_by_side():
    # A 90-column table is 3 chunks; with two small tables, 5 calls of 0.2 s: a second at 1, about 0.4 s at 4.
    requests = [("wide", _facts("w", 3 * CHUNK_COLUMNS), ""), ("a", _facts("a", 5), ""), ("b", _facts("b", 5), "")]
    started = time.monotonic()
    annotations = LLMEnricher(_Model(delay=0.2)).annotate_tables(requests, concurrency=4)
    assert time.monotonic() - started < 0.7
    assert [len(a.columns) for a in annotations] == [90, 5, 5]


def test_a_failing_endpoint_stops_only_that_tables_remaining_chunks():
    started = time.monotonic()
    annotations = LLMEnricher(_Model(fail_after=3)).annotate_tables(_REQUESTS, concurrency=2)
    assert time.monotonic() - started < 5, "the run ends rather than hangs"
    assert len(annotations) == 3
    described = sum(len(a.columns) for a in annotations)
    assert 0 < described < 120, "what was answered is kept, and the rest is not described"
    assert all(a.failures for a in annotations if len(a.columns) < {"wide": 70, "small": 10, "mid": 40}[a.table])


def test_the_client_counts_every_call_from_several_threads(monkeypatch):
    # LLMClient adds each response to running totals; from several workers at once an unlocked `+=` loses updates.
    from types import SimpleNamespace

    from mnemiq.config import Settings
    from mnemiq.llm.client import LLMClient

    client = LLMClient(Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="x", llm_model="m"))
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    monkeypatch.setattr(sys, "getswitchinterval", sys.getswitchinterval)
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        def count():
            for _ in range(2000):
                client._count(response)

        workers = [threading.Thread(target=count) for _ in range(8)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        sys.setswitchinterval(previous)
    assert (client.calls, client.prompt_tokens, client.completion_tokens) == (16000, 48000, 32000)


def _snapshot(tmp_path):
    from mnemiq.adapters.sqlite import SQLiteAdapter
    from mnemiq.enrichment.pipeline import enrich_structural

    path = tmp_path / "s.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE a (" + ", ".join(f"a{i} TEXT" for i in range(45)) + ");"
                      "CREATE TABLE b (b0 TEXT, b1 TEXT); CREATE TABLE c (" + ", ".join(f"c{i} TEXT" for i in range(31)) + ");")
    con.close()
    return enrich_structural(SQLiteAdapter(str(path)), "src"), str(path)


def test_enrich_semantic_snapshot_is_the_same_at_one_and_four(tmp_path):
    from mnemiq.enrichment.pipeline import content_version
    from mnemiq.enrichment.semantic import enrich_semantic

    snap, _ = _snapshot(tmp_path)
    one = enrich_semantic(snap, LLMEnricher(_Model()), concurrency=1)
    four = enrich_semantic(snap, LLMEnricher(_Model()), concurrency=4)
    assert content_version(one) == content_version(four)
    assert sum(1 for c in four.columns if c.description) == 78


def test_the_cli_describes_with_enrich_concurrency(monkeypatch, tmp_path):
    from mnemiq.adapters.sqlite import SQLiteAdapter
    from mnemiq.cli import _cmd_enrich
    from mnemiq.config import Settings

    _, path = _snapshot(tmp_path)
    seen = {}

    class _Stop(Exception):
        pass

    def described(snap, enricher, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr("mnemiq.adapters.resolve.adapter_for", lambda *a, **k: SQLiteAdapter(path))
    monkeypatch.setattr("mnemiq.enrichment.semantic.enrich_semantic", described)
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "src", "kind": "sqlite", "target": path, "catalog": "src",
                                     "schema": "main"}]))
    settings = Settings(sources_path=str(manifest), store_path=str(tmp_path / "store.duckdb"),
                        llm_base_url="http://127.0.0.1:9/v1", llm_api_key="x", llm_model="m", enrich_concurrency=3)
    with pytest.raises(_Stop):
        _cmd_enrich(settings)
    assert seen.get("concurrency") == 3
    assert Settings(sources_path=str(manifest)).enrich_concurrency == 4
