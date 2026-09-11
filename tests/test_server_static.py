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


def test_the_workbench_mirrors_every_deferral_reason():
    """The enum is Python's and the workbench keeps a copy, so a new code drifts silently until
    an operator meets it. Measured twice now: M35's `undefined_term` fell through to "the engine
    declined without a recognised reason code" in exactly the case a code had just been added
    for, and `verifier_unavailable` did the same under an aria-label of "Source failure".

    The check lives HERE, on the producer's side, because that is the side that changes first.
    A workbench test cannot fail for a code that does not exist in it yet -- `DeferralCard.test`
    builds its cases from `Object.keys(REASONS)`, so it enumerates the copy and never the
    original.

    Both files, because they go stale independently: the union in `types.ts` is what TypeScript
    checks, and the `REASONS` map in `verdict.ts` is what the card actually reads.
    """
    import pathlib
    import re

    from mnemiq.contract.seams import DeferralReason

    root = pathlib.Path(__file__).resolve().parents[1] / "workbench" / "src" / "lib"
    union = (root / "types.ts").read_text()
    reasons = (root / "verdict.ts").read_text()

    block = re.search(r"export type DeferralReason =\n((?:\s*\|\s*\"[a-z_]+\"\n?)+)", union)
    assert block, "the DeferralReason union could not be parsed -- this check just went blind"
    declared = set(re.findall(r'"([a-z_]+)"', block.group(1)))
    mapped = set(re.findall(r"^  ([a-z_]+): \{$", reasons, re.M))

    # Controls. Without them a regex that stops matching passes this test on two empty sets.
    assert len(declared) >= 8, f"parsed only {declared} from the union"
    assert len(mapped) >= 8, f"parsed only {mapped} from REASONS"

    expected = {c.value for c in DeferralReason}
    assert declared == expected, f"types.ts union drifted: {expected ^ declared}"
    assert mapped == expected, f"verdict.ts REASONS drifted: {expected ^ mapped}"
