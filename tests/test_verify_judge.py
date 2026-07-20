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
