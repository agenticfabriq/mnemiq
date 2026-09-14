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


def test_process_db_runs_every_case_once_across_workers():
    # concurrency plumbing: with >1 worker, every case is processed exactly once and
    # results come back complete. Fakes stand in for the LLM engine and the DB.
    from mnemiq.eval.bird_runner import _process_db

    cases = [_case(i, "shop", "simple") for i in range(10)]

    class _FakeClient:
        total_tokens = 7
        calls = 1

    class _FakeAdapter:
        def execute(self, sql):
            return [(1,)]  # _gold_too_big probe: 1 row, under any cap

        def execute_arrow(self, sql, timeout_s=None):
            return pa.table({"n": [5]})  # gold and candidate both -> match

    def build_engine_fn():
        def ask(_q):
            return AgentAnswer(answer="5", trace=_trace(), deferred=False)
        adapter = _FakeAdapter()  # one fake serves engine + gold (both return the same table)
        return ask, adapter, adapter, _FakeClient()

    out, clients = _process_db(cases, build_engine_fn, max_rows_cap=1000, workers=3)

    results = [r for kind, r in out if kind == "result"]
    assert len(out) == 10
    assert all(r.outcome is Outcome.CORRECT for r in results)
    assert {r.case_id for r in results} == {f"bird-{i}" for i in range(10)}
    assert clients  # at least one per-worker client collected for token totals


def test_process_db_sequential_path_matches():
    from mnemiq.eval.bird_runner import _process_db

    cases = [_case(1, "shop", "simple")]

    class _A:
        def execute(self, sql):
            return [(1,)]

        def execute_arrow(self, sql, timeout_s=None):
            return pa.table({"n": [5]})

    def build():
        a = _A()  # engine + gold adapter (same fake)
        return (lambda _q: AgentAnswer(answer="5", trace=_trace(), deferred=False)), a, a, None

    out, _ = _process_db(cases, build, max_rows_cap=1000, workers=1)
    assert len(out) == 1 and out[0][0] == "result"


def test_process_db_threads_engine_and_gold_adapters():
    import pyarrow as pa

    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.contract import EvaluationCase, IdentityContext, Trace
    from mnemiq.eval.bird_runner import _process_db
    from mnemiq.eval.harness import Outcome

    def _answer(_q):
        trace = Trace(question="q", plan_sql="CANDIDATE", target_sql="CANDIDATE",
                      result_shape="scalar", timing={"total_ms": 1.0}, enrichment_version="v1",
                      identity=IdentityContext(tenant_id="t", principal_id="u"),
                      tables_used=["person"])
        return AgentAnswer(answer="820.", trace=trace, deferred=False)

    class _Engine:  # runs the candidate SQL only
        def execute(self, sql):
            return [(0,)]  # _gold_too_big probe -> not too big

        def execute_arrow(self, sql, timeout_s=None):
            assert sql == "CANDIDATE"
            return pa.table({"total": [820]})

    class _Gold:  # runs the gold SQL, and the candidate for the portability check
        def execute(self, sql):
            return [(0,)]

        def execute_arrow(self, sql, timeout_s=None):
            if sql == "GOLD":
                return pa.table({"n": [820]})
            assert sql == "CANDIDATE"
            return pa.table({"total": [820]})

    class _Client:
        total_tokens = 0
        calls = 0

    case = EvaluationCase(id="c1", question="q", gold_sql="GOLD", answerable=True, db_id="d")
    out, clients = _process_db(
        [case], lambda: (_answer, _Engine(), _Gold(), _Client()), max_rows_cap=1000, workers=1
    )
    assert [k for k, _ in out] == ["result"]
    assert out[0][1].outcome is Outcome.CORRECT
    assert out[0][1].portable_to_gold_engine is True


def test_enrich_bird_db_cache_key_reflects_phase_toggles(monkeypatch):
    from mnemiq.config import Settings
    from mnemiq.eval.bird_runner import _enrich_cache_suffix

    monkeypatch.setenv("MNEMIQ_ENRICH_FACTS", "1")
    monkeypatch.setenv("MNEMIQ_ENRICH_EXAMPLES", "0")
    assert _enrich_cache_suffix(Settings.from_env()) == "__facts"
    monkeypatch.setenv("MNEMIQ_ENRICH_EXAMPLES", "1")
    assert _enrich_cache_suffix(Settings.from_env()) == "__facts_examples"
    monkeypatch.setenv("MNEMIQ_ENRICH_FACTS", "0")
    monkeypatch.setenv("MNEMIQ_ENRICH_EXAMPLES", "0")
    assert _enrich_cache_suffix(Settings.from_env()) == ""


# ---------------------------------------------------------------------------
# M105: `_run_grouped` and `_process_db` pass the benchmark's duplicate-row
# declaration down to `run_case`. Every other fixture in this file returns gold
# and candidate at the SAME multiplicity, so `dupes_ok` cannot change an
# asserted outcome there and the pass-through would be free to be dropped.
# ---------------------------------------------------------------------------


def _dupe_build(db_id):
    """Gold repeats a row; the candidate states each distinct row once."""
    adapter = _Adapter({"n": [1, 1, 2]}, {"n": [1, 2]})

    def ask(_q):
        return AgentAnswer(answer="1,2", trace=_trace(), deferred=False)

    return ask, adapter


def _dupe_engine_fn():
    """The shape `_process_db` builds: (ask, engine_adapter, gold_adapter, client)."""

    class _A:
        def execute(self, sql):
            return [(1,)]  # _gold_too_big probe: under any cap

        def execute_arrow(self, sql, timeout_s=None):
            return pa.table({"n": [1, 1, 2]} if sql == "GOLD" else {"n": [1, 2]})

    def ask(_q):
        return AgentAnswer(answer="1,2", trace=_trace(), deferred=False)

    class _Client:
        total_tokens = 7
        calls = 1

    a = _A()
    return ask, a, a, _Client()


def test_process_db_defaults_to_the_multiset_reading():
    """`_process_db` is the helper both production runners dispatch through -- `run_bird` and
    `run_minidev_pg` -- while `_run_grouped` is reached only from this file.

    What this does NOT reach is the join above it: deleting the argument from either
    runner's `_process_db(...)` call leaves all of these green. That link is unpinned, and
    naming it is cheaper than a test that would have to build a whole run."""
    from mnemiq.eval.bird_runner import _process_db

    case = _case(1, "shop", "simple").model_copy(update={"gold_sql": "GOLD"})
    out, _ = _process_db([case], _dupe_engine_fn, max_rows_cap=1000, workers=1)
    results = [r for kind, r in out if kind == "result"]
    assert results[0].outcome is Outcome.WRONG, "collapsed without a declaration"


def test_process_db_passes_a_declaration_through_to_the_grader():
    from mnemiq.eval.bird_runner import _process_db

    case = _case(1, "shop", "simple").model_copy(update={"gold_sql": "GOLD"})
    out, _ = _process_db([case], _dupe_engine_fn, max_rows_cap=1000, workers=1,
                         dupes_ok=True)
    results = [r for kind, r in out if kind == "result"]
    assert results[0].outcome is Outcome.CORRECT, "the declaration did not reach run_case"


def test_run_grouped_passes_a_declaration_through_to_the_grader():
    """The sibling path, kept because `_run_grouped` takes the same argument and would
    otherwise be the one place it could silently stop being threaded."""
    results = _run_grouped([_case(1, "shop", "simple")], _dupe_build, gold_sql_sentinel="GOLD")
    assert results[0].outcome is Outcome.WRONG, "collapsed without a declaration"
    results = _run_grouped([_case(1, "shop", "simple")], _dupe_build,
                           gold_sql_sentinel="GOLD", dupes_ok=True)
    assert results[0].outcome is Outcome.CORRECT, "the declaration did not reach run_case"


def test_a_BIRD_only_runner_defaults_to_BIRDs_rule():
    """`run_minidev_pg` is mini-dev on Postgres and has no second benchmark to serve, so
    omitting the declaration must not quietly produce a number understated against the
    leaderboard. `run_bird` is the opposite case -- shared with Spider 1.0 -- and its default
    stays False on purpose. Pinning both, because the asymmetry is the design and a reader
    who flips either one to 'make them consistent' should go red.
    """
    import inspect

    from mnemiq.eval.bird_runner import run_bird
    from mnemiq.eval.minidev_pg import run_minidev_pg

    pg = inspect.signature(run_minidev_pg).parameters["duplicate_rows_insignificant"]
    assert pg.default is True, "a BIRD-only runner must default to BIRD's rule"

    shared = inspect.signature(run_bird).parameters["duplicate_rows_insignificant"]
    assert shared.default is False, "a runner shared with Spider cannot assume a benchmark"


def test_the_results_meta_records_which_rule_graded_the_run():
    """A resumable file can span a rule change -- `_load_done` restores earlier outcomes
    verbatim -- and `source_rev` only survives for the last segment. Without this the
    artifact cannot say which reading produced its numbers, which is the thing M105 asks for.
    """
    import json
    import tempfile

    from mnemiq.eval.bird_runner import _meta_path, _save_meta

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/results.jsonl"
        _save_meta(path, 1, 1, [], True)
        assert json.load(open(_meta_path(path)))["duplicate_rows_insignificant"] is True
        _save_meta(path, 1, 1, [])
        assert json.load(open(_meta_path(path)))["duplicate_rows_insignificant"] is None, \
            "an unstated rule must not read as `false`"


# ---------------------------------------------------------------------------
# M105: a resumable file can span a change of grading rule. `_load_done`
# restores earlier outcomes verbatim and the meta is rewritten whole, so the
# artifact would claim ONE rule over rows decided two ways.
# ---------------------------------------------------------------------------


def _meta_with(tmp, rule):
    from mnemiq.eval.bird_runner import _save_meta

    path = f"{tmp}/results.jsonl"
    _save_meta(path, 0, 0, [], rule)
    return path


def test_resuming_under_the_same_rule_is_fine():
    import tempfile

    from mnemiq.eval.bird_runner import assert_grading_rule_unchanged

    with tempfile.TemporaryDirectory() as d:
        assert_grading_rule_unchanged(_meta_with(d, True), True, restored=5)


def test_resuming_under_a_DIFFERENT_rule_refuses():
    import tempfile

    import pytest

    from mnemiq.eval.bird_runner import MixedGradingRules, assert_grading_rule_unchanged

    with tempfile.TemporaryDirectory() as d:
        path = _meta_with(d, False)
        with pytest.raises(MixedGradingRules) as exc:
            assert_grading_rule_unchanged(path, True, restored=5)
    msg = str(exc.value)
    assert "mix two rules" in msg
    # Tie the deletion target to the RESULTS path, and say the meta must NOT be named.
    # Two traps here, both hit: "delete" plus "ignored" was satisfied by advice naming the
    # meta instead, and so was `f"delete {path}"` alone -- the meta path is the results path
    # plus a suffix, so it prefix-matches. Either wording sends the operator to the state
    # this very guard refuses on: results with no metadata file at all.
    from mnemiq.eval.bird_runner import _meta_path

    assert f"delete {path}" in msg, "the advice does not name the results file"
    assert _meta_path(path) not in msg, "the advice names the meta file as a deletion target"
    assert "replaced" in msg, "the advice does not say what becomes of the leftover meta"
    assert "BOTH" not in msg, "still telling operators to delete a file that no longer matters"


def test_a_file_that_never_stated_its_rule_cannot_be_confirmed():
    """`None` is not a match. A pre-M105 file's rows are unknowable either way, and assuming
    the convenient answer is the absence-versus-failure collapse this repo keeps paying for."""
    import tempfile

    import pytest

    from mnemiq.eval.bird_runner import MixedGradingRules, assert_grading_rule_unchanged

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(MixedGradingRules) as exc:
            assert_grading_rule_unchanged(_meta_with(d, None), True, restored=5)
    assert "not recorded" in str(exc.value)


def test_nothing_restored_means_nothing_to_mix():
    """A stale meta beside an empty or absent results file must not block a fresh run."""
    import tempfile

    from mnemiq.eval.bird_runner import assert_grading_rule_unchanged

    with tempfile.TemporaryDirectory() as d:
        assert_grading_rule_unchanged(_meta_with(d, False), True, restored=0)


def test_the_operator_can_override_deliberately():
    import os
    import tempfile

    from mnemiq.eval.bird_runner import assert_grading_rule_unchanged

    with tempfile.TemporaryDirectory() as d:
        path = _meta_with(d, False)
        os.environ["MNEMIQ_ALLOW_MIXED_GRADING"] = "1"
        try:
            assert_grading_rule_unchanged(path, True, restored=5)
        finally:
            os.environ.pop("MNEMIQ_ALLOW_MIXED_GRADING", None)


def test_a_results_file_with_no_meta_at_all_cannot_be_confirmed():
    """Results are appended per case and the meta is written after, so a run killed in
    between leaves results with no meta; deleting the meta by hand lands here too.

    Returning early on a missing meta let the guard go silent on exactly that file.
    """
    import tempfile

    import pytest

    from mnemiq.eval.bird_runner import MixedGradingRules, assert_grading_rule_unchanged

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(MixedGradingRules) as exc:
            assert_grading_rule_unchanged(f"{d}/results.jsonl", True, restored=5)
    assert "no metadata file at all" in str(exc.value)


def test_an_exclusion_only_run_is_not_mistaken_for_a_fresh_one():
    """The state emptiness could not see. An EXCLUDED case appends nothing to the results
    file -- it lands only in the meta's `excluded` -- so a run whose progress so far is
    entirely exclusions has real state in the meta and zero result rows, which looks exactly
    like a fresh run beside a leftover meta.

    Reading it as fresh threw the state away: measured, two exclusions and 500 tokens lost,
    and those cases would then be re-probed against the gold to rediscover they are too big.
    """
    import tempfile

    from mnemiq.eval.bird_runner import _meta_is_orphaned, _save_meta

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/results.jsonl"

        # Written beside zero rows, and zero restored: the same run, mid-flight.
        _save_meta(path, 500, 3, ["bird-4"], True, results_rows=0)
        assert not _meta_is_orphaned(path, restored=0), "exclusion-only progress read as stale"

        # Written beside one row and none restored: the results file was deleted.
        _save_meta(path, 500, 3, ["bird-4"], True, results_rows=1)
        assert _meta_is_orphaned(path, restored=0), "a deleted results file read as live"

        # An older meta records no count, and unknown is not a match.
        _save_meta(path, 500, 3, ["bird-4"], True)
        assert _meta_is_orphaned(path, restored=0)

        # A meta LAGGING the file is the ordinary mid-flight state, not staleness: results
        # are appended per case while the meta is written at checkpoints. Reading this as
        # `!=` discarded the last checkpoint's totals and exclusions on every resume of a
        # run killed between a row and a checkpoint.
        _save_meta(path, 500, 3, ["bird-4"], True, results_rows=5)
        assert not _meta_is_orphaned(path, restored=7), "a lagging meta read as stale"
        assert _meta_is_orphaned(path, restored=4), "rows disappeared and it read as live"


def test_an_exclusion_only_run_keeps_its_state_when_the_rule_changes():
    """Discarding state and stamping the rule are separate decisions.

    An exclusion-only run's `excluded` entries are gold-too-big verdicts that owe nothing to
    the grading rule, so they survive a change of rule. The rule itself must still be
    restamped -- nothing has been graded under the old one, and leaving it would have the
    meta claim a rule the run does not use.
    """
    import json
    import tempfile

    from mnemiq.eval.bird_runner import _meta_path, _save_meta, resume_state

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/results.jsonl"
        _save_meta(path, 500, 3, ["bird-4"], True, results_rows=0)

        done, tokens, calls, excluded = resume_state(path, False)
        assert (done, tokens, calls, excluded) == ({}, 500, 3, ["bird-4"]), \
            "an exclusion-only run lost its state to a rule change"
        assert json.load(open(_meta_path(path)))["duplicate_rows_insignificant"] is False, \
            "the meta still claims a rule this run does not use"


def test_resume_state_is_the_seam_all_three_runners_share():
    """The JOIN, which is where this went wrong three times.

    `_load_meta` is correct on its own and pinned on its own, and passing it a literal
    instead of `len(done)` restored the stale-metadata bug with the whole suite green. The
    three runners each had the same four lines; now they call this, and this holds it.

    Walked as a sequence because the states only make sense in order.
    """
    import json
    import os
    import tempfile

    import pytest

    from mnemiq.eval.bird_runner import (
        MixedGradingRules,
        _append_result,
        _save_meta,
        resume_state,
    )

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/results.jsonl"
        meta = f"{path}.meta.json"
        row = lambda cid: _mk(cid, Outcome.CORRECT, "simple", "shop")  # noqa: E731

        # 1. A real resume under the same rule: results and totals both come back.
        _append_result(path, row("bird-1"))
        _save_meta(path, 9999, 42, ["bird-7"], True, results_rows=1)
        done, tokens, calls, excluded = resume_state(path, True)
        assert set(done) == {"bird-1"}
        assert (tokens, calls, excluded) == (9999, 42, ["bird-7"])

        # 2. Starting over: results deleted, meta left behind. Nothing carries over -- not
        #    the totals, and above all not the exclusions, which would shrink the denominator.
        os.remove(path)
        assert resume_state(path, True) == ({}, 0, 0, []), "a fresh run inherited a stale meta"

        # 3. The stale meta is REPLACED, not merely skipped, and the rule is stamped before
        #    the first row. Skipping alone is temporary -- see `_claim_meta`.
        claimed = json.load(open(meta))
        assert claimed["excluded"] == [] and claimed["tokens"] == 0, "stale meta survived"
        assert claimed["duplicate_rows_insignificant"] is True, "the rule was not claimed"
        assert claimed["results_rows"] == 0, "the claim must record the rows it stands beside"

        # 4. That fresh run answers one case and is killed. Resuming must stay clean AND must
        #    not be refused: without the claim in step 3 these rows would have no recorded
        #    rule, which `assert_grading_rule_unchanged` rejects as unconfirmable.
        _append_result(path, row("bird-2"))
        done, tokens, calls, excluded = resume_state(path, True)
        assert set(done) == {"bird-2"}
        assert (tokens, calls, excluded) == (0, 0, []), "the stale meta reattached on resume"

        # 5. Resuming THOSE rows under the other rule refuses.
        with pytest.raises(MixedGradingRules):
            resume_state(path, False)

        # 6. But starting over under the other rule is allowed -- it is what the refusal's
        #    own advice says to do. This pins `len(done)` for the REFUSAL, which steps 1-5 do
        #    not: a nonzero literal there locks the operator out with no escape but
        #    MNEMIQ_ALLOW_MIXED_GRADING=1.
        os.remove(path)
        assert resume_state(path, False) == ({}, 0, 0, []), \
            "refused a fresh run that restored nothing"
        assert json.load(open(meta))["duplicate_rows_insignificant"] is False


def test_resume_state_handles_a_run_with_no_results_path():
    """Every run without `--results`, and nothing else covered it.

    `resume_state` has two `if results_path` guards and BOTH are load-bearing now. Measured
    by deleting each in turn: the `_load_done` one raises `os.path.isfile(None)` TypeError,
    and the `_load_meta` one raises on `None + str` when it builds a meta path, since
    `_load_meta` no longer carries a restored-count early return to stop short of it.

    That second half used to be belt-and-braces and this docstring said so. Moving the
    staleness decision out of `_load_meta` inverted it, which is why the claim is
    re-measured here rather than carried."""
    from mnemiq.eval.bird_runner import resume_state

    assert resume_state(None, True) == ({}, 0, 0, [])


def test_a_fresh_results_path_gets_its_rule_recorded_before_the_first_row():
    """No meta at all is the state a brand-new `--results` path starts in, and no test
    entered `resume_state` that way -- every other one writes a meta by hand first.

    It matters because `run_bird` checkpoints only after a whole db group: without the rule
    on disk from the start, a single-db run, or a kill inside the first group, appends rows
    whose rule was never recorded and cannot be resumed without deleting them.
    """
    import json
    import os
    import tempfile

    from mnemiq.eval.bird_runner import _append_result, _meta_path, resume_state

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/results.jsonl"
        assert resume_state(path, True) == ({}, 0, 0, [])
        assert os.path.isfile(_meta_path(path)), "a fresh run recorded no rule"
        m = json.load(open(_meta_path(path)))
        assert m["duplicate_rows_insignificant"] is True and m["results_rows"] == 0

        # And the row it then appends is resumable rather than unconfirmable.
        _append_result(path, _mk("bird-1", Outcome.CORRECT, "simple", "shop"))
        assert resume_state(path, True)[1:] == (0, 0, [])


def test_every_runner_checkpoint_records_the_row_count_it_stands_beside():
    """`_meta_is_orphaned` reads a missing `results_rows` as orphaned, so a checkpoint that
    omits it makes every resume of that runner wipe the meta -- which is exactly what the
    BIRD runner did until this was caught.

    Asserted against the source because no test drives these runners: `run_bird`,
    `run_minidev_pg` and `run_spider2` have no callers in the suite, so deleting the argument
    from any of them otherwise goes unnoticed.
    """
    import inspect
    import re

    from mnemiq.eval import bird_runner, minidev_pg, spider2

    for mod in (bird_runner, minidev_pg, spider2):
        src = inspect.getsource(mod)
        checkpoints = re.findall(r"_save_meta\(results_path, tokens, calls, excluded.*?\)",
                                 src, re.S)
        assert checkpoints, f"{mod.__name__}: no checkpoint call found -- has it been renamed?"
        for call in checkpoints:
            assert "results_rows=" in call, \
                f"{mod.__name__} checkpoints without results_rows: {call}"


def test_inheriting_exclusion_only_state_is_announced_not_silent():
    """The one case the files cannot decide, so the operator has to be told.

    A meta with state and no result rows is an exclusion-only run mid-flight AND a finished
    one someone may have meant to restart, and nothing on disk separates them -- an excluded
    case writes no row, so there is no results file to delete as a signal. Inheriting is the
    right default; inheriting SILENTLY is what hides the wrong case, because `excluded` is
    computed against `max_rows_cap` and a re-run under a different cap would skip those cases
    without ever probing them.
    """
    import tempfile
    import warnings

    from mnemiq.eval.bird_runner import _save_meta, resume_state

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/results.jsonl"
        _save_meta(path, 500, 3, ["bird-4"], True, results_rows=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert resume_state(path, True)[1:] == (500, 3, ["bird-4"])
        assert caught, "inherited prior state without saying so"
        msg = str(caught[0].message)
        assert "fresh --results path" in msg, "no route to starting clean"
        assert "max-rows" in msg, "does not flag that exclusions were capped elsewhere"

        # And no noise when there is nothing to inherit.
        _save_meta(path, 0, 0, [], True, results_rows=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resume_state(path, True)
        assert not caught, "warned about inheriting nothing"
