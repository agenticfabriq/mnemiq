from __future__ import annotations

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
from mnemiq.verify.judge import SemanticJudge
from mnemiq.verify.verifier import Verifier

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
    verify: bool = False,
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

    verifier = None
    if verify or settings.verify:
        judge = None
        if settings.verify_judge:
            base, key = settings.verify_endpoint()
            judge = SemanticJudge(LLMClient(settings.model_copy(update={
                "llm_base_url": base, "llm_api_key": key,
                "llm_model": settings.verify_model or settings.llm_model})))
        verifier = Verifier(threshold=settings.verify_threshold, sanity=settings.verify_sanity,
                            grounding=settings.verify_grounding, judge=judge)

    agent = Agent(
        # Generate in the source's dialect (BIRD: SQLite), so nothing needs a cross-dialect
        # transpile the SQLite writer can't do (DuckDB YEAR()/EXTRACT -> strftime).
        generator=LLMGenerator(client, dialect=getattr(adapter, "dialect", "duckdb"),
                               guided_sql=settings.guided_sql, assertive=settings.assertive_sql),
        synthesizer=LLMSynthesizer(client),
        adapter=adapter,
        cache=TwoTierCache(L1Cache()),
        budget=Budget(wall_clock_s=120.0),
        candidates=candidates,
        corrector=LLMCorrector(client),
        values=ValueIndex(con),
        selector=LLMSelector(client) if candidates > 1 else None,
        verifier=verifier,
    )

    # k=12 measured +2.3 strict / +2.3 facts over k=6 (recall probe: k=12 -> 100% gold-table
    # coverage; k=6 left 48 cases, mostly big DBs, without their gold table). Tunable per source.
    k = settings.retrieval_k

    def ask(question: str) -> AgentAnswer:
        packet = retrieve(con, question, IDENTITY, authz, embedder, k=k,
                          definitions=definitions, table_facts=snapshot.table_facts)
        return agent.answer(packet, snapshot, grants, IDENTITY)

    return ask, client
