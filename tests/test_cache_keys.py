from mnemiq.cache.keys import cache_key, canonical_plan


def test_formatting_does_not_fork_the_cache():
    a = canonical_plan("select  a,b   from claim")
    b = canonical_plan("SELECT a, b FROM claim")
    assert a == b


def test_a_different_query_is_a_different_plan():
    assert canonical_plan("SELECT a FROM claim") != canonical_plan("SELECT b FROM claim")


def test_unparseable_sql_still_yields_a_stable_plan():
    # never raise on the cache path: a cache miss is fine, a crash is not
    assert canonical_plan("!! not sql") == canonical_plan("!! not sql")


def test_identical_grants_share_an_entry():
    a = cache_key("SELECT a FROM claim", "fp-analyst", "v1")
    b = cache_key("SELECT a FROM claim", "fp-analyst", "v1")
    assert a == b  # this is where the hit rate comes from


def test_different_grants_can_never_collide():
    # the whole point: a result computed under broad access must never be served to narrow
    broad = cache_key("SELECT a FROM claim", "fp-admin", "v1")
    narrow = cache_key("SELECT a FROM claim", "fp-intern", "v1")
    assert broad != narrow


def test_re_enrichment_invalidates_everything():
    old = cache_key("SELECT a FROM claim", "fp-analyst", "v1")
    new = cache_key("SELECT a FROM claim", "fp-analyst", "v2")
    assert old != new  # stale answers expire by construction, not by someone noticing


def test_enrich_cache_suffix_separates_ontology_runs():
    from mnemiq.config import Settings
    from mnemiq.eval.bird_runner import _enrich_cache_suffix

    plain = _enrich_cache_suffix(Settings())
    onto = _enrich_cache_suffix(Settings(ontology_records_path="/x/a.json"))
    other = _enrich_cache_suffix(Settings(ontology_records_path="/x/b.json"))
    assert plain != onto and onto != other


def test_enrich_cache_suffix_separates_certified_record_sets():
    from mnemiq.config import Settings
    from mnemiq.eval.bird_runner import _enrich_cache_suffix

    s = Settings(verity_records_url="https://v/api/semantic/records")
    assert _enrich_cache_suffix(s, certified_digest="") == _enrich_cache_suffix(s)
    assert _enrich_cache_suffix(s, certified_digest="aaa") != _enrich_cache_suffix(s, certified_digest="bbb")
