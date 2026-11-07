"""A small real deployment for engine tests whose model is the scripted server in `tests/sweep/stub_model.py`.

The store, the source, the access policy and the profiles file are real, and `build_runtime` builds over them; the model
is reached over HTTP through the engine's own client, so what the client counts and what it raises are the real ones.
Only retrieval is replaced, by one that returns the source's one table: the real one needs an index this store has not
built, and neither the tokens counted nor the kind of a model error depends on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

import mnemiq.runtime as runtime_module
from mnemiq.config import Settings
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import save_snapshot

IDENTITY = IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])
# What the scripted writer answers, over the source's one table and its one row.
SQL = "SELECT COUNT(*) AS n FROM wide"


def deployment(tmp_path: Path, url: str, *, profiles: str | None = None, **settings) -> Settings:
    """Settings for a deployment whose model, judge and embedder are all at `url`, with `profiles` as its profiles
    file when given. Each other keyword is a setting."""
    source = tmp_path / "source.duckdb"
    con = duckdb.connect(str(source))
    con.execute("CREATE TABLE wide (col_0 TEXT)")
    con.execute("INSERT INTO wide VALUES ('one')")
    con.close()
    store = tmp_path / "store.duckdb"
    con = init_store(str(store))
    save_snapshot(con, Snapshot(version="v1", source_id="only", created_at="t",
                                columns=[Column(id="wide.col_0", object_id="wide", name="col_0",
                                                data_type="VARCHAR")]))
    con.close()
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{"id": "only", "kind": "duckdb", "target": str(source), "catalog": "src",
                                     "schema": "main"}]))
    authz = tmp_path / "authz.json"
    authz.write_text(json.dumps({"roles": {"analyst": ["wide"]}}))
    if profiles is not None:
        (tmp_path / "profiles.toml").write_text(profiles)
        settings["profiles_path"] = str(tmp_path / "profiles.toml")
    return Settings(acme_data_dir=None, llm_base_url=url, llm_api_key="stub-key", llm_model="stub-model",
                    sources_path=str(manifest), store_path=str(store), authz_path=str(authz),
                    answer_log_path="off", llm_window_check=False, **settings)


def retrieve_the_one_table(monkeypatch) -> None:
    """Retrieval returns the card of the source's one table, whatever is asked."""
    def retrieve(con, question, identity, authz, embedder, **kwargs):
        return ContextPacket(question=question, cards=[RetrievedCard(object_id="wide", card="wide (col_0 TEXT)",
                                                                     score=1.0)],
                             grant_fingerprint="fp", enrichment_version=None)

    monkeypatch.setattr(runtime_module, "retrieve", retrieve)
