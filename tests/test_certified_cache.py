"""M18 — the certified set must survive the sync.

The pull is incremental; `enrich` rebuilds the snapshot from scratch. So before this, a second
`enrich` overlaid an empty delta onto a fresh profile and silently lost every certified record --
along with the `_protected` set that stops the LLM re-guessing the columns a human certified.

Measured live against Keycloak-authenticated Verity before any of this was written: pull 1 returned
38 records and pull 2 returned 0.

Each test here fails against the pre-M18 code.
"""

import io
import json as _json
import time


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _record(object_id, definition="d"):
    return {
        "envelope": {"object_type": "definition", "object_id": object_id, "version": "v1",
                     "source_system": "pg"},
        "payload": {"id": object_id, "term": object_id, "domain": "ops",
                    "definition": definition},
    }


def _settings(tmp_path, **extra):
    from mnemiq.config import Settings
    return Settings(verity_records_url="https://v/api/semantic/records/open",
                    verity_watermark_path=str(tmp_path / "cache.json"), **extra)


def _ids(records):
    return sorted(r.envelope.object_id for r in records)


def _serve(monkeypatch, pages):
    """Serve one response per call, recording the URLs asked for."""
    from mnemiq.enrichment import certified as mod
    seen = []
    remaining = list(pages)

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        payload = remaining.pop(0) if remaining else {"records": [], "next_cursor": None}
        if isinstance(payload, Exception):
            raise payload
        return _Resp(_json.dumps(payload).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_a_second_pull_returns_the_same_set_not_an_empty_delta(tmp_path, monkeypatch):
    """The headline. Two pulls against an unchanged server: 38 then 38, never 38 then 0."""
    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path)
    _serve(monkeypatch, [
        {"records": [_record("a"), _record("b")], "watermark": "t1", "next_cursor": None},
        {"records": [], "watermark": None, "next_cursor": None},   # the empty delta
    ])

    first = fetch_certified_records(settings).records
    second = fetch_certified_records(settings).records

    assert _ids(first) == ["a", "b"]
    assert _ids(second) == ["a", "b"], (
        "the second pull returned a delta and the cached set was lost -- this is M18, and in "
        "production it also empties `_protected` so the LLM re-guesses certified columns"
    )


def test_a_delta_adds_to_the_cached_set(tmp_path, monkeypatch):
    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path)
    _serve(monkeypatch, [
        {"records": [_record("a")], "watermark": "t1", "next_cursor": None},
        {"records": [_record("b")], "watermark": "t2", "next_cursor": None},
    ])

    fetch_certified_records(settings).records
    merged = fetch_certified_records(settings).records

    assert _ids(merged) == ["a", "b"], "a delta must add to what is cached, not replace it"


def test_a_changed_record_replaces_rather_than_duplicates(tmp_path, monkeypatch):
    """Identity is (object_type, object_id). A re-certified record arrives with the same identity
    and new content -- Verity's D125 was the mirror of this defect on its own side."""
    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path)
    # `b` rides along so this discriminates: with no merge the second pull is just the delta, which
    # happens to be one record with the new content -- and the assertion would pass having proved
    # nothing. The untouched record is what makes replacement distinguishable from replacement-of-
    # everything.
    _serve(monkeypatch, [
        {"records": [_record("a", "old"), _record("b")], "watermark": "t1", "next_cursor": None},
        {"records": [_record("a", "new")], "watermark": "t2", "next_cursor": None},
    ])

    fetch_certified_records(settings).records
    merged = fetch_certified_records(settings).records

    assert _ids(merged) == ["a", "b"], f"expected a replaced and b kept, got {_ids(merged)}"
    assert [r for r in merged if r.envelope.object_id == "a"][0].payload.definition == "new"


def test_an_outage_serves_the_cached_set_rather_than_nothing(tmp_path, monkeypatch):
    """Fail-soft used to mean 'lose every certified record'. The cache is what makes it mean what
    it says: a Verity outage degrades to last-known-good."""
    import urllib.error

    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path)
    _serve(monkeypatch, [
        {"records": [_record("a"), _record("b")], "watermark": "t1", "next_cursor": None},
        urllib.error.URLError("verity is down"),
    ])

    fetch_certified_records(settings).records
    during_outage = fetch_certified_records(settings).records

    assert _ids(during_outage) == ["a", "b"], "an outage must not empty the certified set"


def test_an_outage_does_not_advance_the_watermark(tmp_path, monkeypatch):
    import json
    import urllib.error

    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path)
    _serve(monkeypatch, [
        {"records": [_record("a")], "watermark": "t1", "next_cursor": None},
        urllib.error.URLError("verity is down"),
    ])

    fetch_certified_records(settings).records
    before = json.loads((tmp_path / "cache.json").read_text())
    fetch_certified_records(settings).records
    after = json.loads((tmp_path / "cache.json").read_text())

    assert before == after, "a failed pull must leave the sidecar untouched"


def test_a_legacy_watermark_only_sidecar_forces_a_full_pull(tmp_path, monkeypatch):
    """The upgrade path. An existing deployment has a watermark and no records; asking for a delta
    against no base is the bug itself, so the watermark is discarded for one full pull."""
    from mnemiq.enrichment.certified import fetch_certified_records

    url = "https://v/api/semantic/records/open"
    (tmp_path / "cache.json").write_text(_json.dumps({url: "unix_epoch_seconds:1"}))

    settings = _settings(tmp_path)
    seen = _serve(monkeypatch, [
        {"records": [_record("a")], "watermark": "t1", "next_cursor": None},
    ])

    records = fetch_certified_records(settings).records

    assert _ids(records) == ["a"]
    assert "since=" not in seen[0], f"a legacy sidecar must not send a watermark it cannot merge into: {seen[0]}"


def test_a_stale_cache_triggers_a_full_resync(tmp_path, monkeypatch):
    """Withdrawals are invisible to a delta -- `latest_certified_versions` drops a deprecated record
    rather than tombstoning it -- so a merged set only learns about them on a full pull. Verity's
    own comment calls this the safety net; not doing it means grounding forever on withdrawn
    meaning."""
    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path, verity_full_resync_after_secs=0)
    seen = _serve(monkeypatch, [
        {"records": [_record("a"), _record("b")], "watermark": "t1", "next_cursor": None},
        {"records": [_record("a")], "watermark": "t2", "next_cursor": None},
    ])

    fetch_certified_records(settings).records
    after = fetch_certified_records(settings).records

    assert "since=" not in seen[1], f"a stale cache must re-sync in full: {seen[1]}"
    assert _ids(after) == ["a"], (
        "a full re-sync replaces the set, which is the only way a withdrawn record leaves"
    )


def test_a_fresh_cache_stays_incremental(tmp_path, monkeypatch):
    """Non-vacuity for the test above: the resync window must not simply be always-on, or the
    local set is pointless and this is option (b) wearing a costume."""
    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path, verity_full_resync_after_secs=86400)
    seen = _serve(monkeypatch, [
        {"records": [_record("a")], "watermark": "t1", "next_cursor": None},
        {"records": [_record("b")], "watermark": "t2", "next_cursor": None},
    ])

    fetch_certified_records(settings).records
    fetch_certified_records(settings).records

    assert "since=" in seen[1], f"a fresh cache should ask only for the delta: {seen[1]}"


def test_an_incremental_pull_does_not_advance_synced_at(tmp_path, monkeypatch):
    """Otherwise every delta renews the window and the full re-sync never happens -- the withdrawal
    gap would then be permanent while looking fixed."""
    import json

    from mnemiq.enrichment.certified import fetch_certified_records

    settings = _settings(tmp_path, verity_full_resync_after_secs=86400)
    _serve(monkeypatch, [
        {"records": [_record("a")], "watermark": "t1", "next_cursor": None},
        {"records": [_record("b")], "watermark": "t2", "next_cursor": None},
    ])

    fetch_certified_records(settings).records
    first = json.loads((tmp_path / "cache.json").read_text())["sources"][
        "https://v/api/semantic/records/open"]["synced_at"]
    time.sleep(0.01)
    fetch_certified_records(settings).records
    second = json.loads((tmp_path / "cache.json").read_text())["sources"][
        "https://v/api/semantic/records/open"]["synced_at"]

    assert first == second, "an incremental pull must not renew the full-resync window"
