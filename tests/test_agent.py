import pyarrow as pa

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.keys import cache_key
from mnemiq.cache.store import L1Cache, TwoTierCache, to_ipc
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.execute.runner import ExecutionError
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_RESULT = pa.table({"n": [7]})


class _FakeAdapter:
    """A database that answers, or misbehaves exactly as instructed."""

    def __init__(self, errors: int = 0):
        self.errors = errors
        self.queries: list[str] = []

    def execute_arrow(self, sql, timeout_s=None):
        self.queries.append(sql)
        if self.errors > 0:
            self.errors -= 1
            raise Exception('column "typo" does not exist')
        return _RESULT

    def execute(self, sql):  # EXPLAIN, inside decide()
        return []


def _snapshot() -> Snapshot:
    return Snapshot(
        version="v1",
        source_id="acme",
        created_at="2026-07-13T00:00:00Z",
        columns=[Column(id="claim.n", object_id="claim", name="n")],
    )


def _packet() -> ContextPacket:
    return ContextPacket(
        question="how many claims?",
        cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
        grant_fingerprint=_GRANTS.fingerprint,
        enrichment_version="v1",
    )


def _agent(replies, adapter=None, cache=None, synth=None, budget=None):
    return Agent(
        generator=FakeGenerator(replies),
        synthesizer=synth or FakeSynthesizer("There are 7 claims."),
        adapter=adapter or _FakeAdapter(),
        cache=cache or TwoTierCache(L1Cache()),
        budget=budget or Budget(),
    )


def _answer(agent):
    return agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)


def test_the_happy_path_returns_an_answer_and_a_trace():
    result = _answer(_agent(['{"sql": "SELECT n FROM claim"}']))

    assert result.answer == "There are 7 claims."
    assert result.deferred is False
    assert result.trace is not None
    assert result.trace.tables_used == ["claim"]
    assert result.trace.timing["total_ms"] > 0


def test_a_deferral_never_touches_the_database():
    adapter = _FakeAdapter()
    result = _answer(_agent(['{"sql": null, "reason": "no payout table"}'], adapter=adapter))

    assert result.deferred is True
    assert "payout" in result.answer
    assert adapter.queries == []  # nothing ran
    assert result.trace is None


def test_an_unauthorized_table_defers_without_executing():
    adapter = _FakeAdapter()
    result = _answer(_agent(['{"sql": "SELECT x FROM person"}'], adapter=adapter))

    assert result.deferred is True
    assert adapter.queries == []


def test_a_decider_refusal_is_still_repaired_inside_plan_query():
    # the two repair loops nest: the decider's own loop must not be disabled by the agent
    agent = _agent(
        ['{"sql": "SELECT * FROM claim"}', '{"sql": "SELECT n FROM claim"}']  # star, then fixed
    )
    result = _answer(agent)
    assert result.deferred is False
    assert result.answer == "There are 7 claims."


def test_an_execution_error_is_repaired_using_the_database_as_the_authority():
    # the AST guard and EXPLAIN both passed; only running it revealed the truth
    adapter = _FakeAdapter(errors=1)
    agent = _agent(
        ['{"sql": "SELECT n FROM claim"}', '{"sql": "SELECT n FROM claim"}'], adapter=adapter
    )
    result = _answer(agent)

    assert result.deferred is False
    assert len(adapter.queries) == 2  # it tried again after the database complained
    assert agent.generator.calls[1] is not None  # with the error as feedback
    assert "typo" in agent.generator.calls[1]


def test_a_second_identical_question_is_served_from_cache():
    adapter = _FakeAdapter()
    cache = TwoTierCache(L1Cache())

    first = _agent(['{"sql": "SELECT n FROM claim"}'], adapter=adapter, cache=cache)
    assert _answer(first).cached is False

    second = _agent(['{"sql": "SELECT n FROM claim"}'], adapter=adapter, cache=cache)
    result = _answer(second)

    assert result.cached is True
    assert len(adapter.queries) == 1  # the database was never asked twice
    assert result.answer == "There are 7 claims."


def test_a_different_grant_set_never_reads_another_identity_s_cache():
    cache = TwoTierCache(L1Cache())
    cache.put(
        cache_key("SELECT n FROM claim LIMIT 1000", "fp-of-someone-else", "v1"),
        to_ipc(pa.table({"n": [999]})),
    )

    adapter = _FakeAdapter()
    result = _answer(_agent(['{"sql": "SELECT n FROM claim"}'], adapter=adapter, cache=cache))

    assert result.cached is False  # the poisoned entry is unreachable
    assert adapter.queries == ["SELECT n FROM claim LIMIT 1000"]


def test_forced_synthesis_answers_from_the_data_already_fetched():
    synth = FakeSynthesizer("Answered from what we had.")
    agent = _agent(
        ['{"sql": "SELECT n FROM claim"}'], synth=synth, budget=Budget(wall_clock_s=0.0)
    )
    result = _answer(agent)

    assert result.deferred is False
    assert synth.calls[0]["forced"] is True  # tools off, answer from the rows in hand
    assert result.answer == "Answered from what we had."


def test_the_budget_bounds_repair_attempts():
    adapter = _FakeAdapter(errors=99)  # the database never cooperates
    agent = _agent(
        ['{"sql": "SELECT n FROM claim"}'] * 9, adapter=adapter, budget=Budget(max_attempts=2)
    )
    result = _answer(agent)

    assert result.deferred is True
    assert len(adapter.queries) == 2  # bounded, then honest
    assert "typo" in result.answer or "could not" in result.answer.lower()


def test_an_empty_result_is_an_answer_not_an_error():
    class _Empty(_FakeAdapter):
        def execute_arrow(self, sql, timeout_s=None):
            self.queries.append(sql)
            return pa.table({"n": pa.array([], type=pa.int64())})

    result = _answer(_agent(['{"sql": "SELECT n FROM claim"}'], adapter=_Empty()))
    assert result.deferred is False
    assert result.trace is not None  # "no rows matched" is a real, traceable answer


def test_the_timeout_message_reaches_the_repair_loop():
    class _Slow(_FakeAdapter):
        def execute_arrow(self, sql, timeout_s=None):
            self.queries.append(sql)
            raise ExecutionError("The query timed out after 0.5s.")

    agent = _agent(
        ['{"sql": "SELECT n FROM claim"}'] * 3, adapter=_Slow(), budget=Budget(max_attempts=2)
    )
    result = _answer(agent)
    assert result.deferred is True
    assert "timed out" in result.answer.lower()
