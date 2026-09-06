from mnemiq.verify.judge import FakeJudge, SemanticJudge


class _Client:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def complete(self, system, user, max_tokens=512):
        self.seen = (system, user)
        return self.reply


def test_parses_confidence():
    assert SemanticJudge(_Client('{"confidence": 0.82}')).score("q", "schema", "SELECT 1", "1") == 0.82


def test_clamps_out_of_range():
    assert SemanticJudge(_Client('{"confidence": 1.7}')).score("q", "s", "x", "p") == 1.0


def test_fails_open_on_unreadable_reply():
    assert SemanticJudge(_Client("garbage")).score("q", "s", "x", "p") == 1.0


def test_fails_open_on_client_error():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("down")

    assert SemanticJudge(Boom()).score("q", "s", "x", "p") == 1.0


def test_prompt_carries_sql_and_result():
    c = _Client('{"confidence": 0.5}')
    SemanticJudge(c).score("How many?", "TABLE t", "SELECT COUNT(*) FROM t", "42")
    assert "SELECT COUNT(*) FROM t" in c.seen[1] and "42" in c.seen[1]


def test_fake_judge_returns_fixed_score():
    assert FakeJudge(0.1).score("q", "s", "x", "p") == 0.1


def test_a_clamped_1_0_and_a_dead_endpoint_are_ONE_VALUE_and_two_counts():
    """The reason these counters exist. `min(1.0, ...)` clamps a real reply to the same number the
    error path returns, so a judge that approved everything and an endpoint that answered nothing
    write byte-identical score caches. No analysis of that file can separate them -- which is how a
    wholly failed sweep was certified as a measurement reporting "0 wrong caught".
    """
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("down")

    approving = SemanticJudge(_Client('{"confidence": 1.0}'))
    dead = SemanticJudge(Boom())
    assert [approving.score("q", "s", "x", "p") for _ in range(4)] == [1.0] * 4
    assert [dead.score("q", "s", "x", "p") for _ in range(4)] == [1.0] * 4

    assert approving.fallbacks == 0, "a real judgement is not a fallback"
    assert dead.fallbacks == 4, "every one of these was the constant, not a judgement"


def test_the_error_path_is_counted():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("down")

    j = SemanticJudge(Boom())
    for _ in range(3):
        j.score("q", "s", "x", "p")
    assert (j.calls, j.errors, j.unparsed, j.fallbacks) == (3, 3, 0, 3)


def test_an_unreadable_reply_is_counted_SEPARATELY_from_an_error():
    """Different causes, different repairs: an outage needs the endpoint back, an unreadable reply
    needs the prompt or the parser looked at. Folding them into one number loses that."""
    j = SemanticJudge(_Client("no json here"))
    for _ in range(2):
        j.score("q", "s", "x", "p")
    assert (j.calls, j.errors, j.unparsed, j.fallbacks) == (2, 0, 2, 2)


def test_a_healthy_judge_records_no_fallbacks():
    """The control. Without it the counters could be hardwired to the case count and still pass."""
    j = SemanticJudge(_Client('{"confidence": 0.25}'))
    scores = [j.score("q", "s", "x", "p") for _ in range(5)]
    assert scores == [0.25] * 5
    assert (j.calls, j.fallbacks) == (5, 0)


def _retrying(judge, **kw):
    # Resolved from THIS file, not the process CWD: `pythonpath` in pyproject covers "." and "src"
    # but not scripts/, and a CWD-relative insert collects only when pytest starts at the repo root.
    import pathlib
    import sys
    scripts = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_verify_replay import _RetryingJudge
    return _RetryingJudge(judge, backoff=0.001, **kw)


def test_a_recovered_failure_is_not_contamination():
    """The measurement path retries where the product fails open. A call that errored and then
    succeeded leaves a REAL judgement in the cache, so refusing on the underlying error would
    reject every sweep against a flaky endpoint even when every case was recovered."""
    class Flaky:
        n = 0
        def complete(self, *a, **k):
            Flaky.n += 1
            if Flaky.n % 2:
                raise RuntimeError("500")
            return '{"confidence": 0.4}'

    j = SemanticJudge(Flaky())
    r = _retrying(j, attempts=4)
    assert [r.score("q", "s", "x", "p") for _ in range(6)] == [0.4] * 6
    assert j.errors == 6, "the underlying failures are still counted"
    assert r.gave_up == 0, "but none is unrecovered, so the sweep is clean"


def test_an_unrecoverable_failure_is_counted_as_given_up():
    class Dead:
        def complete(self, *a, **k):
            raise RuntimeError("500")

    j = SemanticJudge(Dead())
    r = _retrying(j, attempts=3)
    assert [r.score("q", "s", "x", "p") for _ in range(4)] == [1.0] * 4
    assert r.gave_up == 4, "every case exhausted its attempts and kept the constant"
    # the CALL count too: without it, a mutation to the attempt loop leaves gave_up right while
    # silently paying for a different number of hosted calls, which on a paid endpoint is the cost
    assert j.calls == 12, "4 cases x 3 attempts"


def test_an_unreadable_reply_is_NOT_retried():
    """It is deterministic for a model that cannot emit the JSON, so retrying buys nothing and
    costs a full round of attempts on every case."""
    class Mute:
        calls = 0
        def complete(self, *a, **k):
            Mute.calls += 1
            return "no json"

    j = SemanticJudge(Mute())
    r = _retrying(j, attempts=4)
    assert r.score("q", "s", "x", "p") == 1.0
    assert Mute.calls == 1, "one call, not four"
    assert j.unparsed == 1

    # ...but it is STILL a fail-open constant, so it counts as unrecovered immediately. Not
    # retried and not forgiven: a judge that answers unreadably every time would otherwise fill
    # the cache with constants and certify with `unrecovered` at zero.
    assert r.gave_up == 1, "an unreadable reply is contamination, not a judgement"


def test_a_judge_that_never_emits_json_cannot_certify():
    """The hole this closes, end to end: every call answers, nothing errors, and every score is
    the constant."""
    class Mute:
        def complete(self, *a, **k):
            return "I think it is fine"

    j = SemanticJudge(Mute())
    r = _retrying(j, attempts=4)
    scores = [r.score("q", "s", "x", "p") for _ in range(20)]
    assert scores == [1.0] * 20
    assert (j.errors, j.unparsed, r.gave_up) == (0, 20, 20)


def test_zero_attempts_is_refused_rather_than_silently_skipping_the_judge():
    """`attempts < 1` makes the retry loop body never execute, so the judge is never called and
    every score is the fail-open constant. It would not certify -- `gave_up` still increments, so
    the gate refuses it as contamination -- but refusing at construction says the true thing: the
    judge was never invoked. A contamination refusal after paying for the run names the wrong
    cause."""
    import pytest

    with pytest.raises(ValueError, match="attempts must be >= 1"):
        _retrying(SemanticJudge(_Client('{"confidence": 0.5}')), attempts=0)
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        _retrying(SemanticJudge(_Client('{"confidence": 0.5}')), attempts=-3)

    # and 1 is legal: no retry, but the judge is still called and still counted
    j = SemanticJudge(_Client("unreadable"))
    r = _retrying(j, attempts=1)
    assert r.score("q", "s", "x", "p") == 1.0
    assert (j.calls, r.gave_up) == (1, 1)


def test_a_failing_call_is_retried_and_backs_off(monkeypatch):
    """What the retry is actually for, now that the outage story is gone.

    An earlier test asserted the window exceeded 45s, on the theory that it had to outlast an
    endpoint outage. It did not: raising the reasoning reserve took endpoint errors from 7, 6 and 4
    per ~490 calls to 0 in 487, so those failures were this code truncating its own requests. There
    is no measured outage length to size against, and a test asserting one would pin a number to a
    story rather than to evidence. What IS worth pinning is the SHAPE: that a failure retries at all, and that successive waits
    grow rather than repeat. And a floor, which the first version of this
    left out: growth alone is satisfied by a backoff of 0.001, whose four attempts finish inside
    ten milliseconds -- hammering an endpoint that has just failed.

    The floor is sized from REQUEST RATE, not from any outage length. That distinction is the whole
    point: the retracted claim needed to know how long a provider stays down, which nothing here
    ever measured, whereas "do not re-ask a failing endpoint more than once a second" needs only a
    view about politeness and holds whatever the provider is doing.
    """
    import pathlib
    import sys

    scripts = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_verify_replay as rvr

    slept: list[float] = []
    monkeypatch.setattr(rvr.time, "sleep", slept.append)
    attempts = _argparse_default_attempts(rvr)
    judge = _AlwaysErrors()
    rvr._RetryingJudge(judge, attempts=attempts).score("q", "s", "SELECT 1", "p")

    assert judge.calls == attempts, "every attempt should have been made"
    assert len(slept) == attempts - 1, "one wait between each pair of attempts"
    assert slept == sorted(slept) and slept[0] < slept[-1], f"waits do not grow: {slept}"
    assert min(slept) >= 1.0, f"retries faster than 1/s hammer a failing endpoint: {slept}"


def _argparse_default_attempts(rvr) -> int:
    """Read `--judge-attempts`'s default off the parser the script actually builds, so the test
    tracks the value sweeps run with rather than a constructor default nothing passes."""
    import argparse
    from unittest.mock import patch

    captured = {}
    real_add = argparse.ArgumentParser.add_argument

    def spy(self, *a, **kw):
        if a and a[0] == "--judge-attempts":
            captured["v"] = kw["default"]
        return real_add(self, *a, **kw)

    with patch.object(argparse.ArgumentParser, "add_argument", spy), patch.object(
            argparse.ArgumentParser, "parse_args", side_effect=SystemExit):
        try:
            rvr.main()
        except SystemExit:
            pass
    assert "v" in captured, "--judge-attempts not found; the flag was renamed"
    return captured["v"]


class _AlwaysErrors:
    """Errors on every call, so the retry loop runs to exhaustion and every sleep is taken."""

    def __init__(self) -> None:
        self.calls = self.errors = self.unparsed = 0

    def read(self, *_args, **_kw):
        from mnemiq.verify.judge import JudgeRead
        self.calls += 1
        self.errors += 1
        return JudgeRead(1.0, fell_open=True, reason="error")

    def score(self, *a, **k) -> float:
        return self.read(*a, **k).score


def test_the_retrying_judge_reports_a_verdict_it_never_got():
    """`Verifier` prefers `read`, so this class must answer it truthfully.

    No answer has ever been mislabelled this way -- `main()` builds its verifiers with no judge and
    `judge_scores` calls `score` directly -- so this is the hole that WOULD open the first time a
    verifier is handed a retrying judge, not something observed. Saying otherwise reads as history.
    """
    import pathlib
    import sys

    scripts = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_verify_replay import _RetryingJudge

    exhausted = _RetryingJudge(_AlwaysErrors(), attempts=2, backoff=0)
    got = exhausted.read("q", "s", "SELECT 1", "p")
    assert got.fell_open is True and got.score == 1.0
    assert exhausted.gave_up == 1

    class _Answers(_AlwaysErrors):
        def read(self, *_a, **_k):
            from mnemiq.verify.judge import JudgeRead
            self.calls += 1
            return JudgeRead(0.4, fell_open=False)

    fine = _RetryingJudge(_Answers(), attempts=2, backoff=0)
    got = fine.read("q", "s", "SELECT 1", "p")
    assert got.fell_open is False and got.score == 0.4


def test_a_recovered_retry_is_not_reported_as_a_missing_verdict():
    """The distinction the wrapper exists for: a call that failed and then succeeded IS a
    judgement, and must not read as an unavailable verifier."""
    import pathlib
    import sys

    scripts = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_verify_replay import _RetryingJudge

    class _FailsOnce(_AlwaysErrors):
        def read(self, *_a, **_k):
            from mnemiq.verify.judge import JudgeRead
            self.calls += 1
            if self.calls == 1:
                self.errors += 1
                return JudgeRead(1.0, fell_open=True, reason="error")
            return JudgeRead(0.3, fell_open=False)

    j = _FailsOnce()
    got = _RetryingJudge(j, attempts=3, backoff=0).read("q", "s", "SELECT 1", "p")
    assert j.errors == 1, "it really did fail once"
    assert got.fell_open is False and got.score == 0.3


def test_one_wrapper_reads_many_cases_and_only_the_failing_ones_are_unavailable():
    """A sweep builds ONE wrapper for every case, so `fell_open` must describe THIS call.

    The version this pins replaced computed it by diffing `gave_up` around the call. That is
    indistinguishable from `gave_up > 0` when every test builds a fresh wrapper and reads once --
    and under one long-lived wrapper the absolute form marks every answer after the first failure
    as unavailable, which is the shape the whole feature exists to prevent.
    """
    import pathlib
    import sys

    scripts = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_verify_replay import _RetryingJudge

    class _FailsFirstCaseOnly:
        def __init__(self):
            self.calls = self.errors = self.unparsed = 0

        def read(self, question, *_a, **_k):
            from mnemiq.verify.judge import JudgeRead
            self.calls += 1
            if question == "doomed":
                self.errors += 1
                return JudgeRead(1.0, fell_open=True, reason="error")
            return JudgeRead(0.4, fell_open=False)

        def score(self, *a, **k):
            return self.read(*a, **k).score

    wrapper = _RetryingJudge(_FailsFirstCaseOnly(), attempts=2, backoff=0)
    first = wrapper.read("doomed", "s", "SELECT 1", "p")
    assert first.fell_open is True and wrapper.gave_up == 1

    for case in ("fine-1", "fine-2", "fine-3"):
        got = wrapper.read(case, "s", "SELECT 1", "p")
        assert got.fell_open is False, f"{case} was judged, on a wrapper that had already given up"
        assert got.score == 0.4
    assert wrapper.gave_up == 1, "no further case gave up"


def test_a_concurrent_failure_on_the_SHARED_inner_judge_does_not_mislabel(monkeypatch):
    """The race one level down. `_RetryingJudge` wraps a judge that is shared, and two earlier
    versions of `read` diffed that judge's `errors`/`unparsed` around their own call -- so another
    thread's failure, landing between the two reads, made this call retry a judgement it already
    had and could end with `fell_open` on an answer that was judged.

    This is the same defect `SemanticJudge.read` was added to retire, which is why it kept coming
    back: it was fixed at the Verifier and left in the wrapper, twice.
    """
    import pathlib
    import sys
    import threading

    scripts = str(pathlib.Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from mnemiq.verify.judge import JudgeRead
    from run_verify_replay import _RetryingJudge

    started = threading.Event()
    may_finish = threading.Event()

    class _SharedInner:
        """One judge behind two wrappers, as a sweep and a server both arrange."""

        def __init__(self):
            self.calls = self.errors = self.unparsed = 0

        def read(self, question, *_a, **_k):
            self.calls += 1
            if question == "judged-but-slow":
                started.set()
                may_finish.wait(2)
                return JudgeRead(0.62, fell_open=False)
            self.errors += 1                      # the OTHER caller's judge fails
            return JudgeRead(1.0, fell_open=True, reason="error")

        def score(self, *a, **k):
            return self.read(*a, **k).score

    inner = _SharedInner()
    out = {}

    def run(q):
        out[q] = _RetryingJudge(inner, attempts=3, backoff=0).read(q, "s", "SELECT 1", "p")

    slow = threading.Thread(target=run, args=("judged-but-slow",))
    other = threading.Thread(target=run, args=("other",))
    slow.start()
    assert started.wait(2)
    other.start()
    other.join()
    may_finish.set()
    slow.join()

    assert out["judged-but-slow"].fell_open is False, out["judged-but-slow"]
    assert out["judged-but-slow"].score == 0.62
    assert out["other"].fell_open is True
