"""Ask a question. Get an answer, the SQL behind it, and the trace."""

from __future__ import annotations

import os
import sys

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import LLMSynthesizer
from mnemiq.authz.grants import DenyAll, FileAuthzProvider, GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.generate.generator import LLMGenerator
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.store.bootstrap import init_store
from mnemiq.store.snapshot_store import current_version, load_snapshot


class _AllowAll:
    """Only for a local demo run with no policy file. Never a default in a real deployment.

    Clears every PII level the snapshot tags, which is what "allow all" has to mean once the
    snapshot reaches retrieval. `GrantSet`'s `pii_clearance` defaults to EMPTY, so without this the
    policy denies every tagged column and the demo silently serves cards with those columns
    stripped -- an answer missing a column, and nothing on screen naming the cause. The eval door
    clears the same set for the same reason.
    """

    def __init__(self, objects, pii_levels=frozenset()):
        self._grants = GrantSet(frozenset(objects), pii_clearance=frozenset(pii_levels))

    def grants_for(self, identity):
        return self._grants


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: answer.py "your question"', file=sys.stderr)
        return 2

    question = sys.argv[1]
    settings = Settings.from_env()
    source_id = os.getenv("MNEMIQ_SOURCE_ID", "acme")
    store_path = os.getenv("MNEMIQ_STORE_PATH", "mnemiq.duckdb")

    con = init_store(store_path)
    version = current_version(con, source_id)
    if version is None:
        print("no snapshot: run enrich_acme.py then build_store.py", file=sys.stderr)
        return 1
    snapshot = load_snapshot(con, version)

    if settings.authz_path:
        authz = FileAuthzProvider(settings.authz_path)
    elif os.getenv("MNEMIQ_ALLOW_ALL") == "1":
        indexed = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]
        authz = _AllowAll(indexed, {c.pii_level for c in snapshot.columns
                                   if c.pii_level and c.pii_level != "none"}
                          if snapshot else frozenset())
        print("WARNING: MNEMIQ_ALLOW_ALL=1 -- every table is visible, and every column: "
              "PII levels the enrichment tagged are CLEARED, so those values can appear "
              "in the SQL and in the printed answer", file=sys.stderr)
    else:
        authz = DenyAll()

    identity = IdentityContext(
        tenant_id="local",
        principal_id=os.getenv("MNEMIQ_PRINCIPAL", "local"),
        roles=[r for r in os.getenv("MNEMIQ_ROLES", "").split(",") if r],
    )

    # See `ask.py`: `retrieve` defaults `definitions` to `()`, so an enabled guard checked
    # declared terms against an empty corpus and refused all of them.
    packet = retrieve(con, question, identity, authz, LLMEmbedder(settings), k=5,
                      definitions=snapshot.definitions if snapshot else (),
                      # The certified measures and the snapshot, for the same reason the
                      # product path passes them: `apply_certified` appends metrics and
                      # dimensions to the snapshot and `retrieve` defaults both to `()`,
                      # so omitting them grounds on less than the snapshot carries --
                      # silently, because an absent argument is not an error (M81).
                      table_facts=snapshot.table_facts if snapshot else (),
                      columns=snapshot.columns if snapshot else (),
                      metrics=snapshot.metrics if snapshot else (),
                      dimensions=snapshot.dimensions if snapshot else (),
                      snapshot=snapshot)
    print(f"retrieved: {[c.object_id for c in packet.cards]}\n")

    agent = Agent(
        # BOTH halves. The Agent alone was the guard wired at one end: with the generator
        # silent the model declares nothing, `ungrounded_terms([])` is `[]`, and the flag
        # reads as on while guarding nothing.
        generator=LLMGenerator(LLMClient(settings),
                               declare_assumed_terms=settings.guard_undefined_terms),
        synthesizer=LLMSynthesizer(LLMClient(settings)),
        adapter=DuckDBPostgresAdapter(settings.pg_dsn),
        cache=TwoTierCache(L1Cache()),
        guard_undefined_terms=settings.guard_undefined_terms,
    )
    result = agent.answer(packet, snapshot, authz.grants_for(identity), identity)

    if result.deferred:
        print(f"DEFERRED: {result.answer}")
        return 0

    print(f"ANSWER: {result.answer}\n")
    print(f"SQL:\n{result.trace.target_sql}\n")
    print(f"tables: {result.trace.tables_used}")
    print(f"timing: {result.trace.timing}")
    print(f"cached: {result.cached}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
