import io
import json
import urllib.error

from mnemiq.config import Settings
from mnemiq.enrichment import verity_auth


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _configured(**overrides) -> Settings:
    base = dict(verity_records_url="https://v/api/semantic/records",
                verity_token_url="https://v/api/auth/token",
                verity_client_id="cid_abc123",
                verity_client_secret="s3cret")
    base.update(overrides)
    return Settings(**base)


def _minting(calls: list, token: str = "tok-1", expires_in: int = 600):
    """urlopen stub that records each request and mints `token`."""
    def fake_urlopen(req, timeout=0):
        calls.append(req)
        return _Resp(json.dumps(
            {"access_token": token, "token_type": "Bearer", "expires_in": expires_in}).encode())
    return fake_urlopen


def test_access_token_posts_client_credentials_and_returns_the_token(monkeypatch):
    verity_auth.reset_token_cache()
    calls: list = []
    monkeypatch.setattr(verity_auth.urllib.request, "urlopen", _minting(calls))

    assert verity_auth.access_token(_configured()) == "tok-1"

    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == "https://v/api/auth/token"
    assert request.get_method() == "POST"
    body = json.loads(request.data)
    assert body["client_id"] == "cid_abc123"
    assert body["client_secret"] == "s3cret"
    assert body["grant_type"] == "client_credentials"


def test_access_token_is_cached_until_it_nears_expiry(monkeypatch):
    verity_auth.reset_token_cache()
    calls: list = []
    monkeypatch.setattr(verity_auth.urllib.request, "urlopen", _minting(calls))
    settings = _configured()

    assert verity_auth.access_token(settings) == "tok-1"
    assert verity_auth.access_token(settings) == "tok-1"
    assert len(calls) == 1, "second call must be served from the cache"


def test_access_token_refetches_once_the_cached_token_expires(monkeypatch):
    verity_auth.reset_token_cache()
    calls: list = []
    # A 30s TTL is under the 60s skew margin, so the entry is never considered fresh.
    monkeypatch.setattr(verity_auth.urllib.request, "urlopen",
                        _minting(calls, token="tok-short", expires_in=30))
    settings = _configured()

    assert verity_auth.access_token(settings) == "tok-short"
    assert verity_auth.access_token(settings) == "tok-short"
    assert len(calls) == 2, "an all-but-expired token must be re-acquired"


def test_force_refresh_bypasses_the_cache(monkeypatch):
    verity_auth.reset_token_cache()
    calls: list = []
    monkeypatch.setattr(verity_auth.urllib.request, "urlopen", _minting(calls))
    settings = _configured()

    assert verity_auth.access_token(settings) == "tok-1"
    assert verity_auth.access_token(settings, force_refresh=True) == "tok-1"
    assert len(calls) == 2


def test_access_token_is_none_when_credentials_are_not_configured(monkeypatch):
    verity_auth.reset_token_cache()
    calls: list = []
    monkeypatch.setattr(verity_auth.urllib.request, "urlopen", _minting(calls))

    # A records URL alone (dev / local-auth-mode Verity) must not attempt an exchange.
    assert verity_auth.access_token(
        Settings(verity_records_url="https://v/api/semantic/records")) is None
    assert verity_auth.access_token(_configured(verity_client_secret=None)) is None
    assert calls == []


def test_access_token_is_fail_soft_on_token_endpoint_failure(monkeypatch):
    verity_auth.reset_token_cache()

    def boom(req, timeout=0):
        raise urllib.error.HTTPError("https://v/api/auth/token", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(verity_auth.urllib.request, "urlopen", boom)
    assert verity_auth.access_token(_configured()) is None


def test_access_token_is_fail_soft_on_a_response_without_a_token(monkeypatch):
    verity_auth.reset_token_cache()
    monkeypatch.setattr(verity_auth.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(json.dumps({"status": "refused"}).encode()))
    assert verity_auth.access_token(_configured()) is None


def test_a_failed_refresh_evicts_the_cached_token(monkeypatch):
    verity_auth.reset_token_cache()
    calls: list = []
    monkeypatch.setattr(verity_auth.urllib.request, "urlopen", _minting(calls))
    settings = _configured()
    assert verity_auth.access_token(settings) == "tok-1"

    def boom(req, timeout=0):
        raise urllib.error.URLError("verity down")

    monkeypatch.setattr(verity_auth.urllib.request, "urlopen", boom)
    assert verity_auth.access_token(settings, force_refresh=True) is None
    # The revoked/failed credential must not leave a stale token behind for the next call.
    assert verity_auth.access_token(settings) is None
