"""The workbench is served by the engine process itself, so the mount must never
shadow the API and an unbuilt workbench must say so rather than 404."""

from fastapi.testclient import TestClient

import mnemiq.server.app as app_module
from mnemiq.contract import IdentityContext
from mnemiq.server.app import build_app


def _identity():
    return IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])


class _RT:
    def scope(self, identity):
        return {"tables": [{"object_id": "claim", "card": "..."}], "starters": []}


def _built(tmp_path):
    (tmp_path / "index.html").write_text("<title>mnemiq workbench</title>")
    (tmp_path / "app.js").write_text("console.log(1)")
    return tmp_path


def test_root_serves_the_built_workbench(tmp_path):
    c = TestClient(build_app(_RT(), _identity(), static_dir=_built(tmp_path)))
    r = c.get("/")
    assert r.status_code == 200
    assert "mnemiq workbench" in r.text


def test_built_assets_are_reachable(tmp_path):
    c = TestClient(build_app(_RT(), _identity(), static_dir=_built(tmp_path)))
    assert c.get("/app.js").status_code == 200


def test_unbuilt_workbench_says_how_to_build_it(tmp_path):
    c = TestClient(build_app(_RT(), _identity(), static_dir=tmp_path / "never-built"))
    r = c.get("/")
    assert r.status_code == 200
    assert "pnpm build" in r.text


def test_defaults_to_the_in_package_bundle_directory(tmp_path, monkeypatch):
    # Must not read the real directory: whether it exists depends on someone having
    # run `pnpm build`, and a test that changes answer with the build is no test.
    monkeypatch.setattr(app_module, "STATIC_DIR", _built(tmp_path))
    r = TestClient(build_app(_RT(), _identity())).get("/")
    assert r.status_code == 200
    assert "mnemiq workbench" in r.text


def test_defaults_to_the_instruction_when_that_directory_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path / "never-built")
    r = TestClient(build_app(_RT(), _identity())).get("/")
    assert r.status_code == 200
    assert "pnpm build" in r.text


def test_the_mount_never_shadows_the_api(tmp_path):
    c = TestClient(build_app(_RT(), _identity(), static_dir=_built(tmp_path)))
    assert c.get("/healthz").json() == {"ok": True}
    assert c.get("/v1/schema").json() == {
        "tables": [{"object_id": "claim", "card": "..."}],
        "starters": [],
    }
