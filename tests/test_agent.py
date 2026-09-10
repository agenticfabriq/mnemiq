import logging

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

    # M6: the source rejecting every attempt is a FAILURE, not the engine deferring. What this
    # test exists to pin -- that the loop is bounded and then honest -- is unchanged.
    assert result.failed is True
    assert result.deferred is False
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
    assert result.failed is True  # M6: a timeout is the source failing us, not us abstaining
    assert "timed out" in result.answer.lower()


# --- Plan 12: self-consistency (candidate voting) -------------------------------------
class _VotingAdapter:
    """Returns a result keyed by a marker in the SQL, so candidates form clusters."""

    def execute(self, sql):  # EXPLAIN inside decide()
        return []

    def execute_arrow(self, sql, timeout_s=None):
        u = sql.upper()
        if "SUM" in u:
            return pa.table({"n": [5]})
        if "MAX" in u:
            return pa.table({"n": [9]})
        return pa.table({"n": [7]})  # count(*) / plain case


def _sql(expr):
    return f'{{"sql": "SELECT {expr} AS n FROM claim", "reason": "ok"}}'


def _vote_agent(replies, candidates, synth="There are 7."):
    return Agent(
        generator=FakeGenerator(replies),
        synthesizer=FakeSynthesizer(synth),
        adapter=_VotingAdapter(),
        cache=TwoTierCache(L1Cache()),
        candidates=candidates,
    )


def test_plurality_result_wins_and_agreement_is_reported():
    # 3 -> 7, one -> 5 (SUM), one -> 9 (MAX): the 7-cluster (3/5) wins
    agent = _vote_agent(
        [_sql("count(*)"), _sql("count(*)"), _sql("count(*)"), _sql("sum(n)"), _sql("max(n)")], 5
    )
    ans = _answer(agent)
    assert not ans.deferred
    assert ans.agreement == 0.6  # 3 of 5
    assert "3/5" in ans.answer
    assert "There are 7." in ans.answer


def test_a_candidate_the_source_refuses_is_dropped_loudly_not_silently(caplog):
    """`_execute` drops a rejected candidate with `return None`, and the vote just gets smaller.

    With 3 of 5 refused and 2 surviving, the answer reports `candidates_executed=2` and reads
    exactly like a narrower vote the engine chose. Self-consistency then decides on a sample
    the SOURCE truncated. The log is the only thing that tells those apart, so it is the log
    this pins -- and it must carry the source's words, since our half is the same sentence for
    every refusal (M95).
    """
    leaky = 'permission denied for table hr_prod.payroll_salary'

    class _RefusesSum(_VotingAdapter):
        def execute_arrow(self, sql, timeout_s=None):
            if "SUM" in sql.upper():
                raise ExecutionError("The source rejected this query.", source_detail=leaky)
            return super().execute_arrow(sql, timeout_s=timeout_s)

    agent = Agent(
        generator=FakeGenerator([_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")]),
        synthesizer=FakeSynthesizer("There are 7."),
        adapter=_RefusesSum(), cache=TwoTierCache(L1Cache()), candidates=3,
    )
    with caplog.at_level(logging.WARNING, logger="mnemiq.agent.loop"):
        ans = _answer(agent)

    assert not ans.deferred
    dropped = [r.getMessage() for r in caplog.records
               if r.getMessage().startswith("a candidate's execution failed")]
    assert len(dropped) == 1, dropped
    assert leaky in dropped[0], "our sentence alone cannot tell one refusal from another"


def test_a_tie_breaks_to_the_earliest_cluster():
    # 5,5,7,7 -> two clusters of 2; the earlier (5-cluster) wins
    agent = _vote_agent(
        [_sql("sum(n)"), _sql("sum(n)"), _sql("count(*)"), _sql("count(*)")], 4, synth="five"
    )
    ans = _answer(agent)
    assert ans.agreement == 0.5
    assert "2/4" in ans.answer


def test_falls_back_to_single_repairing_path_when_no_candidate_is_valid():
    defers = ['{"sql": null, "reason": "cannot"}'] * 3
    agent = _vote_agent(defers + [_sql("count(*)")], 3, synth="fallback answer")
    ans = _answer(agent)
    assert not ans.deferred
    assert ans.agreement is None  # single-path answer carries no agreement
    assert "fallback answer" in ans.answer


def test_single_candidate_mode_is_unchanged():
    agent = _vote_agent([_sql("count(*)")], 1, synth="one")
    ans = _answer(agent)
    assert ans.agreement is None
    assert ans.answer == "one"  # no note appended


def test_agent_passes_its_corrector_into_plan_query(monkeypatch):
    import mnemiq.agent.loop as loop
    from mnemiq.generate.correct import FakeCorrector
    from mnemiq.sql.verdict import Approved

    seen = {}

    def fake_plan_query(packet, snapshot, grants, generator, **kwargs):
        seen["corrector"] = kwargs.get("corrector")
        return Approved(plan_sql="SELECT 1", target_sql="SELECT 1", tables=[], columns=[])

    monkeypatch.setattr(loop, "plan_query", fake_plan_query)
    corrector = FakeCorrector([])
    agent = Agent(
        generator=FakeGenerator(['{"sql":"SELECT 1"}']),
        synthesizer=FakeSynthesizer("ok"),
        adapter=_VotingAdapter(),
        cache=TwoTierCache(L1Cache()),
        corrector=corrector,
    )
    agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)
    assert seen["corrector"] is corrector


def test_agent_passes_its_values_into_plan_query(monkeypatch):
    import mnemiq.agent.loop as loop
    from mnemiq.sql.verdict import Approved

    seen = {}

    def fake_plan_query(packet, snapshot, grants, generator, **kwargs):
        seen["values"] = kwargs.get("values")
        return Approved(plan_sql="SELECT 1", target_sql="SELECT 1", tables=[], columns=[])

    monkeypatch.setattr(loop, "plan_query", fake_plan_query)
    sentinel = object()
    agent = Agent(
        generator=FakeGenerator(['{"sql":"SELECT 1"}']),
        synthesizer=FakeSynthesizer("ok"),
        adapter=_FakeAdapter(),
        cache=TwoTierCache(L1Cache()),
        values=sentinel,
    )
    agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)
    assert seen["values"] is sentinel


# --- Plan 16: selector-judge + strategy diversity ---------------------------------------
def test_candidates_cycle_the_three_strategies():
    from mnemiq.agent.loop import STRATEGIES

    agent = _vote_agent([_sql("count(*)")] * 5, 5)
    _answer(agent)
    assert agent.generator.strategies == list(STRATEGIES) + ["direct", "decompose"]


def test_unanimous_clusters_never_consult_the_selector():
    from mnemiq.execute.select import FakeSelector

    selector = FakeSelector([0])
    agent = _vote_agent([_sql("count(*)")] * 3, 3)
    agent.selector = selector
    ans = _answer(agent)
    assert not ans.deferred
    assert selector.calls == []  # single cluster: selection is vacuous
    assert ans.judge_engaged is False
    assert ans.judge_override is False


def test_disagreement_engages_the_selector_and_its_pick_wins():
    from mnemiq.execute.select import FakeSelector

    # 2x count(*) -> 7, 1x sum -> 5: majority is the 7-cluster (index 0);
    # the judge overrides to the 5-cluster (index 1)
    selector = FakeSelector([1])
    agent = _vote_agent([_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")], 3, synth="five")
    agent.selector = selector
    ans = _answer(agent)

    assert len(selector.calls) == 1
    question, views = selector.calls[0]
    assert [v.size for v in views] == [2, 1]
    assert "SELECT" in views[0].sql and "rows" in views[0].preview
    assert ans.judge_engaged is True
    assert ans.judge_override is True
    assert ans.agreement == 1 / 3  # agreement reflects the CHOSEN cluster


def test_selector_agreeing_with_majority_is_not_an_override():
    from mnemiq.execute.select import FakeSelector

    selector = FakeSelector([0])
    agent = _vote_agent([_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")], 3)
    agent.selector = selector
    ans = _answer(agent)
    assert ans.judge_engaged is True
    assert ans.judge_override is False
    assert ans.agreement == 2 / 3


def test_without_a_selector_voting_behavior_is_unchanged():
    agent = _vote_agent(
        [_sql("count(*)"), _sql("count(*)"), _sql("count(*)"), _sql("sum(n)"), _sql("max(n)")], 5
    )
    ans = _answer(agent)
    assert ans.agreement == 0.6  # exactly Plan 12's pick
    assert ans.judge_engaged is None  # no selector wired: telemetry stays None


# --- Plan 17: selective answering (the min_agreement gate) ------------------------------
def test_fragmented_candidates_defer_and_never_consult_the_judge():
    from mnemiq.execute.select import FakeSelector

    selector = FakeSelector([0])
    agent = _vote_agent([_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")], 3)
    agent.selector = selector
    agent.min_agreement = 1.0
    ans = _answer(agent)

    assert ans.deferred is True
    assert "disagreed" in ans.answer and "2 of 3" in ans.answer
    assert selector.calls == []  # gate fires before the judge
    assert ans.agreement == 2 / 3
    assert ans.candidates_executed == 3


def test_full_set_unanimity_passes_the_gate():
    agent = _vote_agent([_sql("count(*)")] * 3, 3)
    agent.min_agreement = 1.0
    ans = _answer(agent)
    assert ans.deferred is False
    assert ans.candidates_executed == 3


def test_reduced_set_unanimity_is_survival_bias_and_defers():
    # one candidate defers -> only 2 of 3 executed; they agree, but 2/2 is not 3/3
    agent = _vote_agent(
        ['{"sql": null, "reason": "cannot"}', _sql("count(*)"), _sql("count(*)")], 3
    )
    agent.min_agreement = 1.0
    ans = _answer(agent)
    assert ans.deferred is True
    assert "2 of 3" in ans.answer
    assert ans.candidates_executed == 2


def test_without_min_agreement_fragmentation_still_answers():
    agent = _vote_agent([_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")], 3)
    ans = _answer(agent)
    assert ans.deferred is False  # min_agreement=None: today's behavior


def test_the_single_path_reports_no_candidate_count():
    ans = _answer(_agent(['{"sql": "SELECT n FROM claim"}']))
    assert ans.candidates_executed is None


def test_agent_transpiles_to_the_adapters_dialect(monkeypatch):
    # The bug: target was hardcoded "duckdb", so DuckDB SQL hit non-DuckDB sources
    # un-transpiled (BIRD's SQLite: "no such function: YEAR"). The Agent must transpile
    # to the dialect the adapter actually executes.
    import mnemiq.agent.loop as loop_mod
    from mnemiq.generate.plan_query import Deferred

    captured = {}

    def fake_plan_query(packet, snapshot, grants, generator, **kw):
        captured["dialect"] = kw.get("dialect")
        captured["target"] = kw.get("target")
        return Deferred(reason="stop")

    monkeypatch.setattr(loop_mod, "plan_query", fake_plan_query)

    class _SqliteAdapter:
        dialect = "sqlite"

        def execute_arrow(self, sql, timeout_s=None):
            return _RESULT

        def execute(self, sql):
            return []

    agent = _agent(["ignored"], adapter=_SqliteAdapter())
    ans = agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)
    assert ans.deferred is True
    # source dialect drives BOTH parsing and execution -- no cross-dialect transpile
    assert captured["dialect"] == "sqlite"
    assert captured["target"] == "sqlite"


def test_agent_defaults_to_duckdb_when_the_adapter_is_silent(monkeypatch):
    import mnemiq.agent.loop as loop_mod
    from mnemiq.generate.plan_query import Deferred

    captured = {}

    def fake_plan_query(packet, snapshot, grants, generator, **kw):
        captured["dialect"] = kw.get("dialect")
        captured["target"] = kw.get("target")
        return Deferred(reason="stop")

    monkeypatch.setattr(loop_mod, "plan_query", fake_plan_query)
    agent = _agent(["ignored"], adapter=_FakeAdapter())  # _FakeAdapter declares no dialect
    agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)
    assert captured["dialect"] == "duckdb"  # the product default, unchanged
    assert captured["target"] == "duckdb"


def test_real_adapters_declare_their_execution_dialect():
    from mnemiq.adapters.duckdb_postgres import DuckDBPostgresAdapter
    from mnemiq.adapters.sqlite import SQLiteAdapter

    assert SQLiteAdapter.dialect == "sqlite"
    assert DuckDBPostgresAdapter.dialect == "duckdb"


def test_an_answer_reports_the_attempt_that_produced_it():
    """M33: the outer loop repairs what the DATABASE rejected, and `instant` caps it at one
    while `thinking` allows three -- a difference nothing on the answer used to record."""
    adapter = _FakeAdapter(errors=1)
    agent = _agent(
        ['{"sql": "SELECT n FROM claim"}', '{"sql": "SELECT n FROM claim"}'], adapter=adapter
    )
    result = _answer(agent)
    assert result.deferred is False
    assert result.attempts == 2, "the first attempt failed in the source; the second answered"


def test_a_first_time_answer_says_one_attempt_and_no_repair():
    result = _answer(_agent(['{"sql": "SELECT n FROM claim"}']))
    assert result.attempts == 1
    assert result.corrected is False
