import pyarrow as pa

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import EvaluationCase, IdentityContext, Trace
from mnemiq.eval.bird_runner import _run_grouped  # the pure routing core, engine injected
from mnemiq.eval.harness import CaseResult, Outcome
from mnemiq.eval.report import slice_by


def _case(qid, db, diff, gold="SELECT count(*) AS n FROM t"):
    return EvaluationCase(id=f"bird-{qid}", question="q", gold_sql=gold, db_id=db, tags=[diff])


def _mk(case_id, outcome, difficulty, db_id):
    return CaseResult(case_id=case_id, outcome=outcome, difficulty=difficulty, db_id=db_id)


class _Adapter:
    def __init__(self, gold_rows, cand_rows):
        self._gold, self._cand = gold_rows, cand_rows

    def execute_arrow(self, sql, timeout_s=None):
        return pa.table(self._gold if sql == "GOLD" else self._cand)


def _trace():
    return Trace(question="q", plan_sql="CAND", target_sql="CAND", result_shape="scalar",
                 timing={"total_ms": 1.0}, enrichment_version="v1",
                 identity=IdentityContext(tenant_id="t", principal_id="u"), tables_used=["t"])


def test_each_case_is_routed_to_its_databases_adapter():
    # two DBs, each with its own adapter and gold; a right answer in each
    cases = [_case(1, "shop", "simple"), _case(2, "bank", "moderate")]

    built = []

    def build(db_id):
        built.append(db_id)
        adapter = _Adapter({"n": [5]}, {"n": [5]})

        def ask(_q):
            return AgentAnswer(answer="5", trace=_trace(), deferred=False)

        return ask, adapter

    results = _run_grouped(cases, build, gold_sql_sentinel="GOLD")
    assert sorted(built) == ["bank", "shop"]  # one build per DB
    assert all(r.outcome is Outcome.CORRECT for r in results)


def test_slice_by_difficulty_and_db():
    results = [
        _mk("bird-1", Outcome.CORRECT, "simple", "shop"),
        _mk("bird-2", Outcome.WRONG, "moderate", "shop"),
        _mk("bird-3", Outcome.CORRECT, "moderate", "bank"),
    ]
    by_diff = slice_by(results, lambda r: r.difficulty)
    assert by_diff["simple"].accuracy == 1.0
    assert by_diff["moderate"].correct == 1 and by_diff["moderate"].wrong == 1
    by_db = slice_by(results, lambda r: r.db_id)
    assert set(by_db) == {"shop", "bank"}


def test_run_case_populates_db_id_and_difficulty():
    # the slices need these on the result; run_case reads them off the case
    from mnemiq.eval.harness import run_case

    case = _case(9, "shop", "moderate")

    class _A:
        def execute_arrow(self, sql, timeout_s=None):
            return pa.table({"n": [5]})

    def ask(_q):
        return AgentAnswer(answer="5", trace=_trace(), deferred=False)

    result = run_case(case, ask, _A())
    assert result.db_id == "shop"
    assert result.difficulty == "moderate"
