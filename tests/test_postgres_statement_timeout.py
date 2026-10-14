"""M152: a model query on a Postgres source is bounded where it runs -- on the server.

`DuckDBAdapter.execute_arrow` bounded a query with DuckDB's interrupt. Measured on a throwaway
Postgres: the interrupt fired, the ROLLBACK it issues waited on the connection still streaming the
scan, and `execute_arrow(..., timeout_s=3)` had not returned after 186 s -- while Postgres ran the
COPY on, still active 18 s after the client process was killed. So the 30 s execution timeout
bounded nothing, and the database kept running every query the engine gave up on.

The serving runtime now attaches each Postgres source with a server-side `statement_timeout` equal
to the execution timeout, merged into the DSN by libpq's own parser; enrich and refresh attach
without one. Measured with the bound: back at 3.0 s, nothing left running, the connection reusable.
"""

from __future__ import annotations

import os
import sqlite3
import time

import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter, attach_literal, bounded_dsn


def _options(dsn: str) -> str:
    from psycopg.conninfo import conninfo_to_dict

    return conninfo_to_dict(dsn).get("options") or ""


def test_a_keyword_dsn_gains_the_bound():
    dsn = bounded_dsn("host=h port=5 user=u password=p dbname=d", 30)
    assert _options(dsn) == "-c statement_timeout=30000"


def test_a_uri_dsn_gains_the_bound_and_keeps_its_parameters():
    from psycopg.conninfo import conninfo_to_dict

    dsn = bounded_dsn("postgresql://u:p@h:5/d?sslmode=require", 2.5)
    parsed = conninfo_to_dict(dsn)
    assert parsed["options"] == "-c statement_timeout=2500" and parsed["sslmode"] == "require"


def test_an_operators_own_options_survive_and_their_timeout_wins():
    """Ours goes first: Postgres applies `-c` settings in order, so an explicit setting of the
    operator's, written later, is the one that holds."""
    dsn = bounded_dsn("host=h options='-c search_path=x -c statement_timeout=0'", 30)
    assert _options(dsn) == "-c statement_timeout=30000 -c search_path=x -c statement_timeout=0"


def test_no_bound_leaves_the_dsn_untouched():
    assert bounded_dsn("host=h dbname=d", None) == "host=h dbname=d"


def test_an_unparseable_dsn_is_refused_without_quoting_it():
    with pytest.raises(ValueError) as caught:
        bounded_dsn("host=h password='unterminated", 30)
    assert "unterminated" not in str(caught.value)


def test_a_quote_in_an_attach_target_no_longer_ends_the_literal(tmp_path):
    """The DSN with the bound carries `options='...'`; any target with a `'` broke the ATTACH."""
    path = tmp_path / "o'brien.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER)")
    con.execute("INSERT INTO t VALUES (7)")
    con.commit()
    con.close()
    assert attach_literal("a'b") == "'a''b'"
    assert DuckDBAdapter.sqlite(str(path)).execute("SELECT id FROM t") == [(7,)]


def test_the_serving_runtime_bounds_its_postgres_source(monkeypatch):
    """The join: `build_runtime` is the serving door, and it must ask for the bound."""
    import mnemiq.runtime as rt_mod
    from mnemiq.adapters import resolve
    from mnemiq.config import Settings
    from mnemiq.contract import Snapshot
    from mnemiq.execute.runner import DEFAULT_TIMEOUT_S

    seen = {}

    def capture(spec, settings, *, read_only=True, statement_timeout_s=None):
        seen["timeout"] = statement_timeout_s
        raise resolve.SourceUnconfigured("stop here")

    snap = Snapshot(version="v1", source_id="s", created_at="t")
    monkeypatch.setattr(rt_mod, "load_current_snapshot", lambda settings, con: (snap, {}))
    monkeypatch.setattr(resolve, "adapter_for", capture)
    settings = Settings(llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k", llm_model="m",
                        pg_dsn="host=h dbname=d", store_path=":memory:")
    with pytest.raises(rt_mod.SnapshotMissing, match="stop here"):
        rt_mod.build_runtime(settings)
    assert seen["timeout"] == DEFAULT_TIMEOUT_S


def test_only_a_postgres_source_is_given_the_bound(monkeypatch):
    from mnemiq.adapters import resolve
    from mnemiq.config import SourceSpec

    seen = []

    class _Capture:
        def __init__(self, **kwargs):
            seen.append((kwargs["attach_type"], kwargs["statement_timeout_s"]))

    monkeypatch.setattr(resolve, "DuckDBAdapter", _Capture)
    for kind in ("postgres", "sqlite"):
        resolve.adapter_for(SourceSpec(id="s", kind=kind, target="x", catalog="src", schema=""),
                            statement_timeout_s=30)
    assert seen == [("POSTGRES", 30), ("SQLITE", None)]


class _RecordingCon:
    def __init__(self):
        self.sql = []

    def execute(self, sql, *args):
        self.sql.append(sql)
        return self


def _attaches(monkeypatch, module, build):
    """The ATTACH statements an adapter issues, on a connection that records instead of
    connecting -- so the bound is checked where it is applied without a live Postgres."""
    con = _RecordingCon()
    monkeypatch.setattr(module, "connect_duckdb", lambda **kw: con)
    monkeypatch.setattr(module, "load_extension", lambda *a, **kw: None)
    monkeypatch.setattr(module, "lock_configuration", lambda *a, **kw: None)
    build()
    return [s for s in con.sql if s.startswith("ATTACH")]


def test_the_single_source_adapter_attaches_postgres_with_the_bound(monkeypatch):
    import mnemiq.adapters.duckdb as module

    (attach,) = _attaches(monkeypatch, module, lambda: DuckDBAdapter.postgres(
        "host=h dbname=d", statement_timeout_s=30))
    assert "options=''-c statement_timeout=30000''" in attach, attach


def test_the_federated_adapter_bounds_each_postgres_catalog_only(monkeypatch):
    import mnemiq.adapters.federated as module
    from mnemiq.config import SourceSpec

    specs = [SourceSpec(id="a", kind="postgres", target="host=h dbname=d", catalog="pg",
                        schema="public"),
             SourceSpec(id="b", kind="sqlite", target="/tmp/x.db", catalog="lite", schema="main")]
    attaches = _attaches(monkeypatch, module,
                         lambda: module.FederatedAdapter(specs, statement_timeout_s=30))
    assert "statement_timeout=30000" in attaches[0] and "AS pg" in attaches[0]
    assert "statement_timeout" not in attaches[1]


@pytest.mark.skipif(not os.getenv("MNEMIQ_PG_DSN"), reason="no Postgres configured")
def test_a_long_statement_is_ended_by_the_server_at_the_bound():
    """Live: the statement the server runs is the thing bounded. pg_sleep(30) through the attached
    connection comes back at the 2 s bound, and the connection answers the next query."""
    adapter = DuckDBAdapter.postgres(os.environ["MNEMIQ_PG_DSN"], statement_timeout_s=2)
    started = time.monotonic()
    with pytest.raises(Exception, match="statement timeout"):
        adapter.execute_arrow("SELECT * FROM postgres_query('src', 'SELECT pg_sleep(30)')",
                              timeout_s=30)
    assert time.monotonic() - started < 10
    assert adapter.execute("SELECT 1") == [(1,)]
