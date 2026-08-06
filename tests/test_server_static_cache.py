"""The bundle's two cache lifetimes.

This is not a micro-optimisation. `index.html` is the only file whose URL survives a
rebuild, so a cached copy pins the browser to the hashes it was built against -- the
engine ships new code and the tab keeps running the old app, with no error anywhere to
say so. That is exactly how it failed once: a rebuilt workbench kept sending the old
request shape because the browser never re-fetched index.html.
"""

from fastapi.testclient import TestClient

from mnemiq.contract import IdentityContext
from mnemiq.server.app import build_app


class _RT:
    def ask(self, *a, **k):  # never reached; these tests only touch static files
        raise AssertionError("static request should not reach the engine")

    def schema(self, _identity):
        return [{"object_id": "claim", "card": "claim(claim_id)"}]


def _client(tmp_path):
    (tmp_path / "index.html").write_text(
        '<!doctype html><script src="/assets/index-abc123.js"></script>'
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index-abc123.js").write_text("console.log(1)")
    (tmp_path / "favicon.svg").write_text("<svg/>")
    app = build_app(_RT(), IdentityContext(tenant_id="t", principal_id="u"), static_dir=tmp_path)
    return TestClient(app)


def test_index_is_revalidated_so_a_rebuild_reaches_the_browser(tmp_path):
    r = _client(tmp_path).get("/")

    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"


def test_hashed_assets_are_immutable(tmp_path):
    r = _client(tmp_path).get("/assets/index-abc123.js")

    assert r.status_code == 200
    assert "immutable" in r.headers["cache-control"]


def test_an_unhashed_file_is_revalidated_too(tmp_path):
    # favicon.svg keeps its name across builds, so it needs the same treatment as index.
    assert _client(tmp_path).get("/favicon.svg").headers["cache-control"] == "no-cache"


def test_revalidation_still_answers_304_when_nothing_changed(tmp_path):
    client = _client(tmp_path)
    first = client.get("/")

    again = client.get("/", headers={"if-none-match": first.headers["etag"]})

    # "no-cache" must not mean "re-send the body every time" -- it means "ask first".
    assert again.status_code == 304


def test_the_api_still_wins_over_the_static_mount(tmp_path):
    # The mount claims "/", so a regression here would shadow every API route.
    r = _client(tmp_path).get("/v1/schema")

    assert r.status_code == 200
    assert r.json()["tables"][0]["object_id"] == "claim"
