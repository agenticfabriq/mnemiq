"""M156: the model calls that read an answer's SQL are shown the statement as asked, never the policy's rewrite.

Measured on the acme_syn demo (2026-10-14): under a row filter, the verifier's judge was shown `plan_sql`, which wraps
the filtered table in the policy's own predicate, and withdrew 9 of 9 answers as west_region against 0 of 6 for an
unfiltered role. The same judge input with only the SQL swapped for the statement as asked scored a median 0.94
against 0.05. The judge's prompt says to check every filter the question implies, and the policy's reads as one it
never implied. Under an entitlements filter the rewrite also names tables and literals the role is not granted, which
no prompt may carry. The selector's cluster views and the synthesizer read the same field.

Each reads `Approved.model_sql()` (#149): the statement as asked in the plan's dialect. With nothing narrowed that is
`plan_sql` itself, so a benchmark, which clears every policy, sends the same bytes as before.
"""
import pyarrow as pa

from mnemiq.agent.loop import Agent
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.execute.select import ClusterView
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.verify.judge import JudgeRead
from mnemiq.verify.verifier import Verifier

_FILTER = "region = 'west'"
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["west_region"])


def _grants(filtered=True):
    return GrantSet(frozenset({"claim"}), row_filters={"claim": _FILTER} if filtered else {})


def _snapshot():
    return Snapshot(version="v1", source_id="acme", created_at="2026-10-14T00:00:00Z",
                    columns=[Column(id="claim.n", object_id="claim", name="n"),
                             Column(id="claim.region", object_id="claim", name="region")])


def _packet(grants):
    return ContextPacket(question="how many claims?", grant_fingerprint=grants.fingerprint, enrichment_version="v1",
                         cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)])


class _Adapter:
    """Answers every statement; a SUM forms its own cluster so the selector is consulted."""

    def __init__(self):
        self.ran = []

    def execute(self, sql):  # EXPLAIN, inside decide()
        return []

    def execute_arrow(self, sql, timeout_s=None):
        self.ran.append(sql)
        return pa.table({"n": [5 if "SUM" in sql.upper() else 7]})


class _Synthesizer(FakeSynthesizer):
    def __init__(self):
        super().__init__("There are 7.")
        self.shown = []

    def answer(self, question, sql, preview, **kw):
        self.shown.append(sql)
        return super().answer(question, sql, preview, **kw)


class _ReadingJudge:
    def __init__(self):
        self.shown = []

    def read(self, question, schema, sql, preview):
        self.shown.append(sql)
        return JudgeRead(0.99, False)


class _ScoringJudge:
    """A judge without `read`, which the verifier takes at its word through `score`."""

    def __init__(self):
        self.shown = []

    def score(self, question, schema, sql, preview):
        self.shown.append(sql)
        return 0.99


class _Selector:
    def __init__(self):
        self.shown: list[ClusterView] = []

    def select(self, question, views):
        self.shown.extend(views)
        return 0


def _sql(expr):
    return f'{{"sql": "SELECT {expr} AS n FROM claim", "reason": "ok"}}'


def _ask(*, filtered=True, verifier=None, selector=None, candidates=1):
    adapter, synth, grants = _Adapter(), _Synthesizer(), _grants(filtered)
    replies = [_sql("count(*)"), _sql("count(*)"), _sql("sum(n)")][:max(candidates, 1)]
    agent = Agent(generator=FakeGenerator(replies), synthesizer=synth, adapter=adapter,
                  cache=TwoTierCache(L1Cache()), candidates=candidates, verifier=verifier)
    if selector is not None:
        agent.selector = selector
    answer = agent.answer(_packet(grants), _snapshot(), grants, _IDENTITY)
    assert not answer.deferred and not answer.failed, answer.answer
    return answer, adapter, synth


def test_the_policy_rewrite_really_runs_in_these_fixtures():
    # The control every test below leans on: the executed statement carries the predicate.
    _, adapter, _ = _ask()
    assert adapter.ran and all("west" in sql for sql in adapter.ran)


def test_the_synthesizer_is_shown_the_statement_as_asked():
    _, _, synth = _ask()
    assert synth.shown == ["SELECT COUNT(*) AS n FROM claim LIMIT 1000"]


def test_the_verifier_s_judge_is_shown_the_statement_as_asked():
    judge = _ReadingJudge()
    _ask(verifier=Verifier(judge=judge))
    assert judge.shown == ["SELECT COUNT(*) AS n FROM claim LIMIT 1000"]


def test_a_judge_without_read_is_shown_the_statement_as_asked_too():
    judge = _ScoringJudge()
    _ask(verifier=Verifier(judge=judge))
    assert judge.shown == ["SELECT COUNT(*) AS n FROM claim LIMIT 1000"]


def test_the_selector_s_cluster_views_show_the_statement_as_asked():
    selector = _Selector()
    _ask(selector=selector, candidates=3)
    assert len(selector.shown) == 2, "two clusters: COUNT and SUM"
    assert all("west" not in view.sql for view in selector.shown)


def test_with_no_policy_every_model_call_sees_the_executed_plan_unchanged():
    # A benchmark clears every policy: what the judge, selector and synthesizer read must stay byte-identical.
    judge, selector = _ReadingJudge(), _Selector()
    _, adapter, synth = _ask(filtered=False, verifier=Verifier(judge=judge), selector=selector, candidates=3)
    executed = set(adapter.ran)
    assert set(synth.shown) <= executed and set(judge.shown) <= executed
    assert {view.sql for view in selector.shown} <= executed
