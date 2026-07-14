import pyarrow as pa

from mnemiq.cache.store import L1Cache, NullCache, TwoTierCache, from_ipc, to_ipc


def _table() -> pa.Table:
    return pa.table({"n": [1, 2, 3], "label": ["a", "b", "c"]})


def test_arrow_survives_the_round_trip_exactly():
    table = _table()
    assert from_ipc(to_ipc(table)).equals(table)


def test_l1_stores_and_returns():
    cache = L1Cache()
    cache.put("k", b"value")
    assert cache.get("k") == b"value"
    assert cache.get("missing") is None


def test_l1_evicts_the_least_recently_used():
    cache = L1Cache(maxsize=2)
    cache.put("a", b"1")
    cache.put("b", b"2")
    cache.get("a")  # a is now the most recently used
    cache.put("c", b"3")  # evicts b

    assert cache.get("a") == b"1"
    assert cache.get("c") == b"3"
    assert cache.get("b") is None


def test_the_null_cache_never_remembers_anything():
    cache = NullCache()
    cache.put("k", b"v")
    assert cache.get("k") is None  # the L2 default: the seam exists, no backend is wired


def test_two_tier_promotes_an_l2_hit_into_l1():
    l1, l2 = L1Cache(), L1Cache()  # a second L1 stands in for a real L2 backend
    l2.put("k", b"value")

    cache = TwoTierCache(l1, l2)
    assert cache.get("k") == b"value"
    assert l1.get("k") == b"value"  # the next read never has to reach L2


def test_two_tier_writes_through_to_both():
    l1, l2 = L1Cache(), L1Cache()
    TwoTierCache(l1, l2).put("k", b"value")
    assert l1.get("k") == b"value" and l2.get("k") == b"value"


def test_two_tier_defaults_to_no_l2():
    cache = TwoTierCache(L1Cache())
    cache.put("k", b"v")
    assert cache.get("k") == b"v"  # L1 alone still works
