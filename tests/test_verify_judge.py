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
    """`attempts < 1` makes the retry loop body never execute, so `score` returns the fail-open
    constant without calling the judge and without counting it -- a whole sweep of constants with
    `unrecovered: 0`, certified, reachable from a command-line flag."""
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
