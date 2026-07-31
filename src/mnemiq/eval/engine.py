from __future__ import annotations

from collections.abc import Callable, Sequence

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent, AgentAnswer
from mnemiq.assembly import build_components
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import Definition, IdentityContext, Snapshot
from mnemiq.llm.client import LLMClient
from mnemiq.llm.embeddings import LLMEmbedder
from mnemiq.semantic.retrieval import retrieve
from mnemiq.semantic.store import build_example_index, build_index
from mnemiq.semantic.values import build_value_index
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

    ontology_index = None
    from mnemiq.enrichment.certified import certified_concept_schemes, fetch_certified_records
    from mnemiq.ontology.records import load_records, merge_records

    _onto = load_records(settings.ontology_records_path) if settings.ontology_records_path else None
    # Governed concept schemes join the local digest so the eval measures the same index production
    # builds. fetch is fail-soft: no verity_records_url -> [] -> no-op (no overhead for local runs).
    _cert_schemes = certified_concept_schemes(fetch_certified_records(settings))
    if _cert_schemes:
        _onto = merge_records(_onto, _cert_schemes)
    if _onto is not None:
        from mnemiq.semantic.ontology_index import OntologyIndex, build_ontology_index

        build_ontology_index(_onto, snapshot, con)
        ontology_index = OntologyIndex(con)

    # The glossary channel has been wired-but-unfed since Plan 08: Snapshot.definitions was
    # never read at runtime. Ontology definitions are the first producer that makes it matter.
    definitions = definitions or snapshot.definitions

    tables = [r[0] for r in con.execute("SELECT object_id FROM semantic_object").fetchall()]
    # Grade SQL capability with FULL data access: clear every PII level the enrichment tagged,
    # so CLS does not refuse legitimate columns (e.g. CustomerID). Governance/RLS/CLS have their
    # own tests; a benchmark that denied enrichment-tagged PII would under-count every attempt.
    levels = frozenset(c.pii_level for c in snapshot.columns if c.pii_level and c.pii_level != "none")
    grants = GrantSet(frozenset(tables), pii_clearance=levels)
    authz = _GrantAll(grants)

    kit = build_components(settings, adapter, con)
    client = kit.client  # one client, so the token count is the run's true cost

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

    # The kit generates in the source's dialect (BIRD: SQLite), so nothing needs a
    # cross-dialect transpile the SQLite writer can't do (DuckDB YEAR()/EXTRACT -> strftime).
    agent = Agent(
        generator=kit.generator,
        synthesizer=kit.synthesizer,
        adapter=adapter,
        cache=TwoTierCache(L1Cache()),
        budget=Budget(wall_clock_s=120.0),
        candidates=candidates,
        corrector=kit.corrector,
        values=kit.values,
        selector=kit.selector if candidates > 1 else None,
        verifier=verifier,
    )

    # k=12 measured +2.3 strict / +2.3 facts over k=6 (recall probe: k=12 -> 100% gold-table
    # coverage; k=6 left 48 cases, mostly big DBs, without their gold table). Tunable per source.
    k = settings.retrieval_k

    def ask(question: str) -> AgentAnswer:
        packet = retrieve(con, question, IDENTITY, authz, embedder, k=k,
                          definitions=definitions, table_facts=snapshot.table_facts,
                          columns=snapshot.columns, ontology_index=ontology_index)
        return agent.answer(packet, snapshot, grants, IDENTITY)

    return ask, client
