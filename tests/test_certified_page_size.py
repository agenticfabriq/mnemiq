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


def test_default_page_size_is_sent_as_limit(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    seen = []

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        return _Resp(_json.dumps({"records": [_record("a")], "watermark": "t1",
                                  "next_cursor": None}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    mod.fetch_certified_records(_settings(tmp_path))  # default page size 500
    assert any("limit=500" in url for url in seen)


def test_bounded_pages_drain_via_next_cursor(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    seen = []

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        # a server that pages while records remain (limit present -> next_cursor set)
        if "cursor=1" in req.full_url:
            return _Resp(_json.dumps({"records": [_record("b")], "watermark": "t2",
                                      "next_cursor": None}).encode())
        return _Resp(_json.dumps({"records": [_record("a")], "watermark": "t2",
                                  "next_cursor": "1"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    records = mod.fetch_certified_records(_settings(tmp_path, verity_page_size=1))
    assert [r.envelope.object_id for r in records] == ["a", "b"]  # drained both pages
    assert all("limit=1" in url for url in seen)  # every page bounded
    assert len(seen) == 2


def test_page_size_zero_omits_limit(tmp_path, monkeypatch):
    from mnemiq.enrichment import certified as mod

    seen = []

    def fake_urlopen(req, timeout=0):
        seen.append(req.full_url)
        return _Resp(_json.dumps({"records": [_record("a")], "watermark": None,
                                  "next_cursor": None}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    mod.fetch_certified_records(_settings(tmp_path, verity_page_size=0))
    assert not any("limit=" in url for url in seen)  # full-dump escape hatch
