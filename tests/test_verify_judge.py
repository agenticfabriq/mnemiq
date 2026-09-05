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
