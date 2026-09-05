"""Ask a question. Get SQL the engine is willing to run -- or an honest deferral."""

from __future__ import annotations

import os
import sys

from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
from mnemiq.authz.grants import DenyAll, FileAuthzProvider, GrantSet
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.generate.generator import LLMGenerator
from mnemiq.generate.plan_query import Deferred, plan_query
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
        print('usage: ask.py "your question"', file=sys.stderr)
        return 2

    question = sys.argv[1]
    settings = Settings.from_env()
    source_id = os.getenv("MNEMIQ_SOURCE_ID", "acme")
    store_path = os.getenv("MNEMIQ_STORE_PATH", "mnemiq.duckdb")

    con = init_store(store_path)
    version = current_version(con, source_id)
    if version is None:
        print(
            "no snapshot: run scripts/enrich_acme.py then scripts/build_store.py",
            file=sys.stderr,
        )
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
        authz = DenyAll()  # no policy is not permission

    identity = IdentityContext(
        tenant_id="local",
        principal_id=os.getenv("MNEMIQ_PRINCIPAL", "local"),
        roles=[r for r in os.getenv("MNEMIQ_ROLES", "").split(",") if r],
    )
    embedder = LLMEmbedder(settings)

    # The certified corpus is the guard's THIRD leg: the prompt asks, the engine checks, and
    # `packet.definitions` is what it checks AGAINST. `retrieve` defaults it to `()`, so with
    # the guard on this script refused every declared term -- including terms the loaded
    # snapshot certifies. Fed the same way the runtime feeds it.
    packet = retrieve(con, question, identity, authz, embedder, k=5,
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
    print(f"retrieved: {[c.object_id for c in packet.cards]}")

    adapter = DuckDBPostgresAdapter(settings.pg_dsn) if settings.pg_dsn else None
    outcome = plan_query(
        packet,
        snapshot,
        authz.grants_for(identity),
        LLMGenerator(LLMClient(settings),
                     declare_assumed_terms=settings.guard_undefined_terms),
        adapter=adapter,
        target="duckdb",
        guard_undefined_terms=settings.guard_undefined_terms,
    )

    if isinstance(outcome, Deferred):
        print(f"\nDEFERRED: {outcome.reason}")
        return 0

    print(f"\ntables: {outcome.tables}")
    print(f"columns: {outcome.columns}")
    print(f"\nSQL:\n{outcome.target_sql}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
