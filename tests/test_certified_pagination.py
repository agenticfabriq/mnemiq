import io
import json as _json


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _record(object_id):
    return {
        "envelope": {"object_type": "definition", "object_id": object_id, "version": "v1",
                     "source_system": "pg"},
        "payload": {"id": object_id, "term": object_id, "domain": "ops", "definition": "d"},
    }


def _settings(tmp_path, **extra):
    from mnemiq.config import Settings
    return Settings(verity_records_url="https://v/api/semantic/records/open",
                    verity_watermark_path=str(tmp_path / "wm.json"), **extra)


def test_drains_all_pages_following_next_cursor(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    seen = []

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        if "cursor=1" in req.full_url:
            return _Resp(_json.dumps({"records": [_record("b")], "watermark": "t2",
                                      "next_cursor": None}).encode())
        return _Resp(_json.dumps({"records": [_record("a")], "watermark": "t2",
                                  "next_cursor": "1"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    records = mod.fetch_certified_records(_settings(tmp_path))
    assert [r.envelope.object_id for r in records] == ["a", "b"]  # both pages
    assert any("cursor=1" in url for url in seen)  # followed next_cursor


def test_persists_watermark_after_full_drain_and_sends_it_next_time(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    def once(req, timeout=0):
        return _Resp(_json.dumps({"records": [_record("a")], "watermark": "t9",
                                  "next_cursor": None}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", once)
    mod.fetch_certified_records(_settings(tmp_path))
    saved = _json.loads((tmp_path / "wm.json").read_text())
    assert saved["https://v/api/semantic/records/open"] == "t9"

    sent = []

    def capture(req, timeout=0):
        sent.append(req.full_url)
        return _Resp(_json.dumps({"records": [], "watermark": "t9", "next_cursor": None}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", capture)
    mod.fetch_certified_records(_settings(tmp_path))
    assert any("since=t9" in url for url in sent)  # incremental on the second pull


def test_partial_drain_does_not_advance_the_watermark(tmp_path, monkeypatch):
    import urllib.error
    from mnemiq.enrichment import certified as mod

    def fake_urlopen(req, timeout=0):
        if "cursor=1" in req.full_url:
            raise urllib.error.URLError("verity blipped mid-drain")
        return _Resp(_json.dumps({"records": [_record("a")], "watermark": "t2",
                                  "next_cursor": "1"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    records = mod.fetch_certified_records(_settings(tmp_path))
    assert [r.envelope.object_id for r in records] == ["a"]  # partial page-1 records, fail-soft
    assert not (tmp_path / "wm.json").exists()  # watermark NOT advanced


def test_missing_sidecar_means_full_pull(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    sent = []

    def fake_urlopen(req, timeout=0):
        sent.append(req.full_url)
        return _Resp(_json.dumps({"records": [_record("a")], "watermark": None,
                                  "next_cursor": None}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    mod.fetch_certified_records(_settings(tmp_path))
    assert not any("since=" in url for url in sent)  # no watermark on disk -> no since=
