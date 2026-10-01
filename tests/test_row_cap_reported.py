"""M118: an answer the guard's row limit cut is said to be cut, not reported as the whole.

The guard appends `LIMIT 1000` to every query that does not bound itself tighter. A query whose
real answer has 11,000 rows came back with 1,000, the preview called that the "true count", the
model was shown "(1000 rows)", and nothing anywhere said the result had been cut. The limit the
query asked for is the query's business; the one the guard imposed is a cut nobody asked for.
"""

import pyarrow as pa

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.eval.harness import Outcome, run_case
from mnemiq.execute.render import render_result
from mnemiq.generate.generator import FakeGenerator
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.guard import MAX_ROWS, check_shape

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])


def test_the_guard_marks_the_limit_it_imposed():
    assert check_shape("SELECT n FROM claim").meta.get("row_cap") == MAX_ROWS
    assert check_shape("SELECT n FROM claim LIMIT 5000").meta.get("row_cap") == MAX_ROWS, "clamped"


def test_a_limit_the_query_set_itself_is_not_a_cut():
    assert check_shape("SELECT n FROM claim LIMIT 10").meta.get("row_cap") is None
    assert check_shape(f"SELECT n FROM claim LIMIT {MAX_ROWS}").meta.get("row_cap") is None


def test_the_render_says_the_result_stopped_at_the_limit():
    rendered = render_result(pa.table({"n": list(range(MAX_ROWS))}), cut_at=MAX_ROWS)
    assert "stopped at the 1,000-row limit" in rendered
    assert f"({MAX_ROWS} rows)" not in rendered and f"of {MAX_ROWS} rows" not in rendered, \
        "a cut result must not be presented as a total"


class _Rows:
    def __init__(self, n: int):
        self.n = n
        self.queries: list[str] = []

    def execute_arrow(self, sql, timeout_s=None):
        self.queries.append(sql)
        return pa.table({"n": list(range(self.n))})

    def execute(self, sql):  # EXPLAIN, inside decide()
        return []


class _Synth:
    def __init__(self):
        self.rendered = None

    def answer(self, question, sql, rendered, forced=False, row_count=None):
        self.rendered = rendered
        return "There are 1,000 claims."


def _ask(sql: str, rows: int):
    synth = _Synth()
    agent = Agent(generator=FakeGenerator([f'{{"sql": "{sql}"}}']), synthesizer=synth,
                  adapter=_Rows(rows), cache=TwoTierCache(L1Cache()), budget=Budget())
    packet = ContextPacket(question="list the claims",
                           cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
                           grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")
    snapshot = Snapshot(version="v1", source_id="acme", created_at="2026-07-13T00:00:00Z",
                        columns=[Column(id="claim.n", object_id="claim", name="n")])
    return agent.answer(packet, snapshot, _GRANTS, _IDENTITY), synth


def test_a_result_the_guard_cut_says_so_in_the_answer_preview_and_prompt():
    answer, synth = _ask("SELECT n FROM claim", rows=MAX_ROWS)
    assert answer.preview.capped is True
    assert "1,000-row limit" in answer.answer, "said after the model, where it cannot be softened"
    assert "stopped at the 1,000-row limit" in synth.rendered


def test_a_result_under_the_limit_is_not_called_cut():
    answer, synth = _ask("SELECT n FROM claim", rows=MAX_ROWS - 1)
    assert answer.preview.capped is False
    assert "row limit" not in answer.answer and "stopped at" not in synth.rendered


def test_a_full_result_under_the_querys_own_limit_is_not_called_cut():
    answer, _ = _ask("SELECT n FROM claim LIMIT 10", rows=10)
    assert answer.preview.capped is False and "row limit" not in answer.answer


def test_the_api_carries_the_flag():
    from mnemiq.server.serialize import answer_payload

    answer, _ = _ask("SELECT n FROM claim", rows=MAX_ROWS)
    assert answer_payload(answer)["preview"]["capped"] is True


def test_an_eval_gold_over_the_limit_is_not_graded_wrong():
    """No answer can pass a gold with more rows than the guard lets through, so grading one
    against it is scoring the cap, not the model."""
    from mnemiq.contract import EvaluationCase

    class _Gold(_Rows):  # the gold query reads 1,140 rows; the candidate, cut by its LIMIT, 1,000
        def execute_arrow(self, sql, timeout_s=None):
            n = MAX_ROWS if "LIMIT" in sql.upper() else MAX_ROWS + 140
            return pa.table({"n": list(range(n))})

    answer, _ = _ask("SELECT n FROM claim", rows=MAX_ROWS)
    case = EvaluationCase(id="big", question="list the claims", gold_sql="SELECT n FROM claim",
                          answerable=True)
    result = run_case(case, lambda q: answer, _Gold(0))
    assert result.outcome == Outcome.ERROR
    assert "1,140" in result.answer and "1,000" in result.answer


def test_under_set_semantics_a_long_gold_of_few_distinct_rows_is_still_graded():
    """BIRD declares duplicates insignificant and the grader collapses the gold to distinct rows:
    1,140 rows of 500 distinct values is matchable by a SELECT DISTINCT, so it must be graded."""
    from mnemiq.contract import EvaluationCase

    class _Repeats(_Rows):  # gold: 1,140 rows over 500 values; the DISTINCT candidate: the 500
        def execute_arrow(self, sql, timeout_s=None):
            if "DISTINCT" in sql.upper():
                return pa.table({"n": list(range(500))})
            return pa.table({"n": [i % 500 for i in range(MAX_ROWS + 140)]})

    answer, _ = _ask("SELECT DISTINCT n FROM claim", rows=500)
    case = EvaluationCase(id="dupes", question="which claim numbers", gold_sql="SELECT n FROM claim",
                          answerable=True)
    graded = run_case(case, lambda q: answer, _Repeats(0), duplicate_rows_insignificant=True)
    assert graded.outcome == Outcome.CORRECT
    multiset = run_case(case, lambda q: answer, _Repeats(0))
    assert multiset.outcome == Outcome.ERROR, "every row counts without the declaration"


def test_mirrored_rows_count_once_as_the_got_facts_reading_counts_them():
    """Got-facts sorts each row's cells before collapsing, so (1, 2) and (2, 1) are one fact: a gold
    of 1,100 distinct ordered pairs that are 550 facts is matchable, and must be graded."""
    from mnemiq.eval.grade import rows_that_count

    pairs = [(i, i + 1) for i in range(550)] + [(i + 1, i) for i in range(550)]
    gold = pa.table({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]})
    assert rows_that_count(gold, duplicate_rows_insignificant=True) == 550
    assert rows_that_count(gold) == 1_100


def test_a_mirrored_gold_over_the_limit_goes_through_run_case_and_is_graded():
    from mnemiq.contract import EvaluationCase

    pairs = [(i, i + 1) for i in range(550)] + [(i + 1, i) for i in range(550)]

    class _Mirrored(_Rows):  # gold: 1,100 ordered pairs; the candidate: each fact once
        def execute_arrow(self, sql, timeout_s=None):
            rows = pairs[:550] if "LIMIT" in sql.upper() else pairs
            return pa.table({"a": [p[0] for p in rows], "b": [p[1] for p in rows]})

    answer, _ = _ask("SELECT n FROM claim", rows=MAX_ROWS)
    case = EvaluationCase(id="mirror", question="which pairs", gold_sql="SELECT a, b FROM pairs",
                          answerable=True)
    graded = run_case(case, lambda q: answer, _Mirrored(0), duplicate_rows_insignificant=True)
    assert graded.outcome == Outcome.CORRECT_FACTS, "graded, and the got-facts match is not lost"
