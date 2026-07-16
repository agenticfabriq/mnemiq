from mnemiq.generate.correct import FakeCorrector, LLMCorrector


class _FakeClient:
    def __init__(self, reply):
        self._reply = reply
        self.seen = {}

    def complete(self, system, user, max_tokens=512):
        self.seen = {"system": system, "user": user}
        return self._reply


def test_llm_corrector_sends_the_problem_and_returns_clean_sql():
    client = _FakeClient(
        "```sql\nSELECT name FROM t WHERE score IS NOT NULL ORDER BY score LIMIT 1\n```"
    )
    out = LLMCorrector(client).correct(
        "SELECT name FROM t ORDER BY score LIMIT 1", "add WHERE score IS NOT NULL"
    )
    assert out == "SELECT name FROM t WHERE score IS NOT NULL ORDER BY score LIMIT 1"
    assert "add WHERE score IS NOT NULL" in client.seen["user"]  # the problem reaches the model
    assert "only" in client.seen["system"].lower()  # constrained: change only what's needed


def test_llm_corrector_handles_a_bare_sql_reply():
    client = _FakeClient("SELECT 1")
    assert LLMCorrector(client).correct("SELECT 2", "p") == "SELECT 1"


def test_fake_corrector_replays_and_records():
    fc = FakeCorrector(["SELECT 1", "SELECT 2"])
    assert fc.correct("x", "p1") == "SELECT 1"
    assert fc.correct("y", "p2") == "SELECT 2"
    assert [c[1] for c in fc.calls] == ["p1", "p2"]
