"""The Superset extension's release: a tag `superset-extension-v<version>` must name the version the extension's
manifest holds, and the release workflow checks it before it packs and publishes the bundle."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "superset-extension-release.yml"


def _check():
    spec = importlib.util.spec_from_file_location("check_extension_release",
                                                  ROOT / "scripts" / "check_extension_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path, version: str = "0.2.0") -> Path:
    path = tmp_path / "extension.json"
    path.write_text(json.dumps({"publisher": "agenticfabriq", "name": "mnemiq", "version": version}))
    return path


def test_a_tag_naming_the_manifests_version_names_its_bundle(tmp_path):
    assert _check().release_bundle("superset-extension-v0.2.0", _manifest(tmp_path)) == "mnemiq-0.2.0.supx"


@pytest.mark.parametrize("tag", ["superset-extension-v0.3.0", "superset-extension-v0.2", "v0.2.0", "0.2.0",
                                 "superset-extension-v0.2.0-rc1"])
def test_a_tag_that_does_not_name_the_manifests_version_is_refused(tmp_path, tag):
    with pytest.raises(ValueError, match="0.2.0"):
        _check().release_bundle(tag, _manifest(tmp_path))


def test_the_repositorys_own_manifest_is_read():
    version = json.loads((ROOT / "integrations" / "superset" / "extension.json").read_text())["version"]
    assert _check().release_bundle(f"superset-extension-v{version}") == f"mnemiq-{version}.supx"


def test_the_extension_s_three_version_pins_agree():
    """A release names the manifest's version; the panel's package and the backend's project carry the same, so a
    release cannot ship one part still at the version before."""
    import tomllib
    superset = ROOT / "integrations" / "superset"
    manifest = json.loads((superset / "extension.json").read_text())["version"]
    panel = json.loads((superset / "frontend" / "package.json").read_text())["version"]
    backend = tomllib.loads((superset / "backend" / "pyproject.toml").read_text())["project"]["version"]
    assert manifest == panel == backend, {"extension.json": manifest, "package.json": panel, "pyproject.toml": backend}


def test_the_workflow_checks_the_tag_before_it_packs_the_frontend_and_publishes():
    """Read from the workflow's text, in order: a tag push publishes with write access to contents, the tag is checked
    first, the panel is built and tested, the bundle is packed only with its frontend, and the release carries it."""
    text = WORKFLOW.read_text()
    assert 'tags: ["superset-extension-v*"]' in text and "contents: write" in text
    order = [text.index(step) for step in ("scripts/check_extension_release.py", "pnpm test", "pnpm build",
                                           "scripts/build_superset_extension.py", "gh release create")]
    assert order == sorted(order), order
    packing = text[text.index("scripts/build_superset_extension.py"):].split("\n", 1)[0]
    assert "--require-frontend" in packing


def test_the_workflow_publishes_only_a_commit_on_main_that_the_tag_still_names():
    """A tag on a commit not on main is refused before anything is built, and the release is made only while the tag,
    as GitHub resolves it, still names the commit built; releases of one tag run one at a time."""
    text = WORKFLOW.read_text()
    assert "fetch-depth: 0" in text and "concurrency:" in text
    on_main = text.index('git merge-base --is-ancestor "$GITHUB_SHA" origin/main')
    tag_still = text.index('"$GITHUB_SHA"', text.index("commits/$TAG"))
    assert on_main < text.index("pnpm install") and tag_still < text.index("gh release create")
