from mnemiq.execute.select import (
    ClusterView,
    FakeSelector,
    LLMSelector,
    MajoritySelector,
    auto_accepted,
    majority_index,
)


def _views(*sizes):
    return [ClusterView(sql=f"SELECT {i}", preview=f"n\n{i}\n(1 rows)", size=s)
            for i, s in enumerate(sizes)]


class _FakeClient:
    def __init__(self, reply):
        self._reply = reply
        self.calls = 0
        self.seen = {}

    def complete(self, system, user, max_tokens=512):
        self.calls += 1
        self.seen = {"system": system, "user": user}
        return self._reply


def test_a_single_cluster_is_auto_accepted():
    assert auto_accepted(_views(5)) is True


def test_two_clusters_are_not_auto_accepted():
    # AUTO_ACCEPT = 1.0: any disagreement engages the judge (spec 3.2 -- at N=5,
    # a 3/5 or 4/5 majority is usually wrong; DeepEye's 0.6 would wave it through)
    assert auto_accepted(_views(4, 1)) is False


def test_majority_index_picks_largest_first_on_ties():
    assert majority_index(_views(2, 3, 1)) == 1
    assert majority_index(_views(2, 2, 1)) == 0  # tie -> earliest, like max(key=len)


def test_majority_selector_returns_the_majority():
    assert MajoritySelector().select("q", _views(1, 3)) == 1


def test_llm_selector_picks_the_indicated_cluster():
    client = _FakeClient('{"choice": 1}')
    idx = LLMSelector(client).select("how many claims?", _views(3, 2))
    assert idx == 1
    assert client.calls == 1
    assert "how many claims?" in client.seen["user"]
    assert "SELECT 0" in client.seen["user"] and "SELECT 1" in client.seen["user"]


def test_llm_selector_reads_a_bare_integer_reply():
    assert LLMSelector(_FakeClient("1")).select("q", _views(3, 2)) == 1


def test_garbage_reply_falls_back_to_majority():
    assert LLMSelector(_FakeClient("no idea")).select("q", _views(2, 3)) == 1


def test_out_of_range_reply_falls_back_to_majority():
    assert LLMSelector(_FakeClient('{"choice": 9}')).select("q", _views(2, 3)) == 1


def test_a_client_exception_falls_back_to_majority():
    class _Boom:
        def complete(self, system, user, max_tokens=512):
            raise RuntimeError("endpoint down")

    assert LLMSelector(_Boom()).select("q", _views(2, 3)) == 1


def test_fake_selector_replays_and_records():
    fs = FakeSelector([1, 0])
    assert fs.select("q1", _views(1, 1)) == 1
    assert fs.select("q2", _views(2, 1)) == 0
    assert [c[0] for c in fs.calls] == ["q1", "q2"]
