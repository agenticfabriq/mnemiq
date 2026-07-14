import os
import sqlite3

import pyarrow as pa

from mnemiq.agent.loop import AgentAnswer
from mnemiq.config import Settings
from mnemiq.contract import EvaluationCase, IdentityContext, Trace
from mnemiq.eval.bird_runner import _run_grouped, enrich_bird_db
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


def _tiny_bird(tmp_path):
    dbdir = tmp_path / "dev_databases" / "toy"
    dbdir.mkdir(parents=True)
    con = sqlite3.connect(dbdir / "toy.sqlite")
    con.executescript("CREATE TABLE t (a INTEGER, b TEXT); INSERT INTO t VALUES (1,'x'),(2,'y');")
    con.commit()
    con.close()
    return str(tmp_path)


def _settings(model):
    return Settings(
        llm_base_url=None, llm_api_key=None, llm_model=model, pg_dsn=None, acme_data_dir=None
    )


def test_enrich_cache_is_model_aware(tmp_path):
    # structural-only, so no LLM; the point is the cache key, not the enrichment
    minidev = _tiny_bird(tmp_path)
    cache = str(tmp_path / "cache")

    enrich_bird_db(minidev, "toy", _settings("openai.gpt-5.5"), cache_dir=cache, semantic=False)
    enrich_bird_db(minidev, "toy", _settings("openai.gpt-5-mini"), cache_dir=cache, semantic=False)

    files = os.listdir(cache)
    # a cache keyed on db_id alone would serve gpt-5-mini's enrichment to a gpt-5.5 run
    assert any("openai.gpt-5.5" in f for f in files)
    assert any("openai.gpt-5-mini" in f for f in files)
    assert len(files) == 2  # two models -> two distinct cache entries


def test_results_checkpoint_round_trips(tmp_path):
    from mnemiq.eval.bird_runner import _append_result, _load_done

    path = str(tmp_path / "results.jsonl")
    r1 = CaseResult(case_id="bird-1", outcome=Outcome.CORRECT, db_id="shop", difficulty="simple")
    r2 = CaseResult(case_id="bird-2", outcome=Outcome.WRONG, db_id="shop", difficulty="moderate")
    _append_result(path, r1)
    _append_result(path, r2)

    done = _load_done(path)
    assert set(done) == {"bird-1", "bird-2"}
    assert done["bird-1"].outcome is Outcome.CORRECT  # reconstructed as the enum, not a string
    assert done["bird-2"].db_id == "shop"


def test_load_done_is_empty_when_no_checkpoint(tmp_path):
    from mnemiq.eval.bird_runner import _load_done

    assert _load_done(str(tmp_path / "missing.jsonl")) == {}


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
