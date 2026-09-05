"""The judge cache must hold judgements only.

A fail-open score is exactly 1.0 and so is an approval, so a constant that reaches the cache can
never be found again by looking at it. That is not a theoretical loss: one transient burst left a
single unrecovered case in a 487-call sweep, and because the constant was indistinguishable from
the 51 genuine 1.0s beside it, the only repair was to re-score all 487.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_verify_replay import _CachingJudge, _RetryingJudge  # noqa: E402


class _Judge:
    """A judge that fails its first `fail_first` calls and answers after that.

    `fail_first=0` always answers; a number larger than the retry budget never does.
    """

    def __init__(self, fail_first: int = 0) -> None:
        self.calls = self.errors = self.unparsed = 0
        self._fail_first = fail_first

    def score(self, question, schema, sql, preview):
        self.calls += 1
        if self.calls <= self._fail_first:
            self.errors += 1
            return 1.0          # the fail-open constant
        return 0.25


def _cache(tmp_path, judge):
    return _CachingJudge(judge, str(tmp_path / "c.json"), "m")


def test_a_score_the_retries_gave_up_on_is_not_written(tmp_path):
    inner = _Judge(fail_first=99)
    c = _cache(tmp_path, _RetryingJudge(inner, attempts=2, backoff=0))
    assert c.score("q", "s", "SELECT 1", "p") == 1.0        # the caller still gets fail-open
    # Not written at all, so not even created. A sweep where EVERY case fails therefore leaves no
    # cache file, which `rerun_local_judge.sh` already refuses on ("the judge never scored") --
    # the same conclusion by a different route, not a hole.
    assert not (tmp_path / "c.json").exists()


def test_a_real_judgement_is_written(tmp_path):
    c = _cache(tmp_path, _RetryingJudge(_Judge(), attempts=2, backoff=0))
    assert c.score("q", "s", "SELECT 1", "p") == 0.25
    assert list(json.load(open(tmp_path / "c.json")).values()) == [0.25]


def test_the_absent_case_is_rescored_by_the_next_run_and_the_rest_are_not(tmp_path):
    """The point of not writing it. The re-run pays for the failures only."""
    path = str(tmp_path / "c.json")
    failing = _RetryingJudge(_Judge(fail_first=99), attempts=2, backoff=0)
    c1 = _CachingJudge(failing, path, "m")
    c1.score("good", "s", "SELECT 1", "p")
    c1.score("bad", "s", "SELECT 2", "p")

    healthy = _Judge()
    c2 = _CachingJudge(_RetryingJudge(healthy, attempts=2, backoff=0), path, "m")
    c2.score("good", "s", "SELECT 1", "p")
    c2.score("bad", "s", "SELECT 2", "p")
    assert healthy.calls == 2      # both, because neither was cached while the judge was failing

    third = _Judge()
    c3 = _CachingJudge(_RetryingJudge(third, attempts=2, backoff=0), path, "m")
    c3.score("good", "s", "SELECT 1", "p")
    c3.score("bad", "s", "SELECT 2", "p")
    assert third.calls == 0        # now both are real judgements and neither is re-paid for


def test_a_judge_without_the_counter_still_caches(tmp_path):
    """`_CachingJudge` wraps whatever it is given; a judge with no `gave_up` must not break it."""
    c = _cache(tmp_path, _Judge())
    assert c.score("q", "s", "SELECT 1", "p") == 0.25
    assert list(json.load(open(tmp_path / "c.json")).values()) == [0.25]


def test_a_score_recovered_on_retry_IS_written(tmp_path):
    """The distinction the fix rests on, and the one the other tests cannot show. A recovered
    retry increments `errors` -- so keying the skip on `errors` would throw away a real
    judgement and re-pay for it on every subsequent run -- while `gave_up` stays put."""
    inner = _Judge(fail_first=1)
    c = _cache(tmp_path, _RetryingJudge(inner, attempts=4, backoff=0))
    assert c.score("q", "s", "SELECT 1", "p") == 0.25
    assert inner.errors == 1                                    # it did fail once
    assert list(json.load(open(tmp_path / "c.json")).values()) == [0.25]
