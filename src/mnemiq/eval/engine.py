from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent, AgentAnswer
from mnemiq.agent.synthesize import LLMSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import Definition, IdentityContext, Snapshot
from mnemiq.execute.select import LLMSelector
from mnemiq.generate.correct import LLMCorrector
from mnemiq.generate.generator import LLMGenerator
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_example_index, build_index
from mnemiq.semantic.values import ValueIndex, build_value_index
from mnemiq.store.bootstrap import init_store

IDENTITY = IdentityContext(tenant_id="local", principal_id="eval", roles=["analyst"])

Engine = Callable[[str], AgentAnswer]


class _GrantAll:
    """The eval grants every indexed table. Authorization has its own tests; what is on
    trial here is whether the engine can answer what it IS allowed to see."""

    def __init__(self, grants: GrantSet) -> None:
        self._grants = grants

    def grants_for(self, _identity: IdentityContext) -> GrantSet:
        return self._grants


def build_engine(
    snapshot: Snapshot,
    adapter,
    settings: Settings,
    store_path: str = ":memory:",
    definitions: Sequence[Definition] = (),
    candidates: int = 1,
) -> tuple[Engine, LLMClient]:
    """Retrieval + agent over one snapshot, assembled exactly once.

    The script and the live test must measure the SAME engine; assembling it twice, two
    subtly different ways, is how an eval drifts away from the thing it claims to measure.
    Returns the ask function and the client whose token counter it shares, so the caller
    can report what the run cost.
    """
    embedder = LLMEmbedder(settings)
    con = init_store(store_path)
    build_index(con, snapshot, embedder)
    build_example_index(con, snapshot, embedder)
    build_value_index(adapter, snapshot, con)

    tables = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]
    # Grade SQL capability with FULL data access: clear every PII level the enrichment tagged,
    # so CLS does not refuse legitimate columns (e.g. CustomerID). Governance/RLS/CLS have their
    # own tests; a benchmark that denied enrichment-tagged PII would under-count every attempt.
    levels = frozenset(c.pii_level for c in snapshot.columns if c.pii_level and c.pii_level != "none")
    grants = GrantSet(frozenset(tables), pii_clearance=levels)
    authz = _GrantAll(grants)

    client = LLMClient(settings)  # one client, so the token count is the run's true cost
    agent = Agent(
        # Generate in the source's dialect (BIRD: SQLite), so nothing needs a cross-dialect
        # transpile the SQLite writer can't do (DuckDB YEAR()/EXTRACT -> strftime).
        generator=LLMGenerator(client, dialect=getattr(adapter, "dialect", "duckdb")),
        synthesizer=LLMSynthesizer(client),
        adapter=adapter,
        cache=TwoTierCache(L1Cache()),
        budget=Budget(wall_clock_s=120.0),
        candidates=candidates,
        corrector=LLMCorrector(client),
        values=ValueIndex(con),
        selector=LLMSelector(client) if candidates > 1 else None,
    )

    # k=12 measured +2.3 strict / +2.3 facts over k=6 (recall probe: k=12 -> 100% gold-table
    # coverage; k=6 left 48 cases, mostly big DBs, without their gold table). Tunable per source.
    k = int(os.getenv("MNEMIQ_RETRIEVAL_K", "12"))

    def ask(question: str) -> AgentAnswer:
        packet = retrieve(con, question, IDENTITY, authz, embedder, k=k,
                          definitions=definitions, table_facts=snapshot.table_facts)
        return agent.answer(packet, snapshot, grants, IDENTITY)

    return ask, client
