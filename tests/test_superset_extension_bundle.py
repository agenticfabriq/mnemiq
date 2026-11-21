"""The extension's packer: a stdlib script that writes `dist/` and a `.supx` holding only the entries Superset
7.0.0rc2's loader accepts. No Superset import."""

import importlib.util
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Copied from superset/extensions/utils.py, `BACKEND_REGEX`; the loader also takes `manifest.json` and
# `FRONTEND_REGEX` (`^frontend/dist/(?!.*\.\.)(.+)$`) entries and refuses every other one.
BACKEND_REGEX = re.compile(r"^backend/src/(?!.*\.\.)(.+)$")
# Copied from the same file, `FRONTEND_REGEX`.
FRONTEND_REGEX = re.compile(r"^frontend/dist/(?!.*\.\.)(.+)$")


def _packer():
    spec = importlib.util.spec_from_file_location("build_superset_extension",
                                                  ROOT / "scripts" / "build_superset_extension.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(tmp_path, *more):
    out = tmp_path / "out"
    assert _packer().main(["--out", str(out), *more]) == 0
    return out


def _dist(tmp_path, *names: str) -> Path:
    """A frontend's built `dist/`, holding a file of each name."""
    dist = tmp_path / "frontend-dist"
    for name in names:
        (dist / name).parent.mkdir(parents=True, exist_ok=True)
        (dist / name).write_text("// built\n")
    return dist


ENTRY = "remoteEntry.0123456789abcdef.js"
CHUNK = "src_index_tsx.89abcdef01234567.js"


def test_frontend_lands_under_frontend_dist(tmp_path):
    out = tmp_path / "out"
    bundle = _packer().build(out, frontend_dist=_dist(tmp_path, ENTRY, CHUNK))
    with zipfile.ZipFile(bundle) as zipped:
        names = zipped.namelist()
    assert f"frontend/dist/{ENTRY}" in names and f"frontend/dist/{CHUNK}" in names
    for name in names:
        assert name == "manifest.json" or BACKEND_REGEX.match(name) or FRONTEND_REGEX.match(name), name
    # the control: the regex refuses what the loader refuses
    assert not FRONTEND_REGEX.match("frontend/src/index.tsx") and not FRONTEND_REGEX.match("frontend/dist/../x")


def test_manifest_names_the_remote_entry_and_container(tmp_path):
    bundle = _packer().build(tmp_path / "out", frontend_dist=_dist(tmp_path, ENTRY, CHUNK))
    with zipfile.ZipFile(bundle) as zipped:
        manifest = json.loads(zipped.read("manifest.json"))
    # rc2's CLI names the container `<publisher>_<name>` in camel case (`get_module_federation_name`).
    assert manifest["frontend"] == {"remoteEntry": ENTRY, "moduleFederationName": "agenticfabriq_mnemiq"}


def test_unhashed_frontend_file_is_refused(tmp_path):
    for bad in ("main.js", "remoteEntry.js", "index.js.map"):
        dist = _dist(tmp_path / bad.replace(".", "_"), ENTRY, bad)
        try:
            _packer().build(tmp_path / "out", frontend_dist=dist)
        except ValueError as refused:
            assert bad in str(refused)
        else:
            raise AssertionError(f"{bad} was packed")


def test_two_remote_entries_are_refused(tmp_path):
    dist = _dist(tmp_path, ENTRY, "remoteEntry.fedcba9876543210.js")
    try:
        _packer().build(tmp_path / "out", frontend_dist=dist)
    except ValueError as refused:
        assert "remote entr" in str(refused)
    else:
        raise AssertionError("two remote entries were packed")


def test_require_frontend_refuses_a_missing_dist(tmp_path):
    try:
        _packer().build(tmp_path / "out", frontend_dist=tmp_path / "nowhere", require_frontend=True)
    except ValueError as refused:
        assert "frontend" in str(refused)
    else:
        raise AssertionError("a missing frontend was packed")


def test_bundle_has_only_allowed_entries(tmp_path):
    out = _build(tmp_path)
    with zipfile.ZipFile(out / "mnemiq-0.2.0.supx") as bundle:
        names = bundle.namelist()
    assert "manifest.json" in names
    assert "backend/src/agenticfabriq/mnemiq/entrypoint.py" in names
    for name in names:
        assert name == "manifest.json" or BACKEND_REGEX.match(name), name
    assert not any(n.endswith("/") or "__pycache__" in n or n.endswith(".pyc") for n in names), names
    # the control: the regex does refuse what the loader refuses
    assert not BACKEND_REGEX.match("backend/pyproject.toml") and not BACKEND_REGEX.match("backend/src/../x")
    # the zip and dist/ agree
    on_disk = {p.relative_to(out / "dist").as_posix() for p in (out / "dist").rglob("*") if p.is_file()}
    assert on_disk == set(names)


def test_manifest_entrypoint(tmp_path):
    out = _build(tmp_path)
    manifest = json.loads((out / "dist" / "manifest.json").read_text())
    assert manifest["publisher"] == "agenticfabriq" and manifest["name"] == "mnemiq"
    assert manifest["id"] == "agenticfabriq.mnemiq" and manifest["version"] == "0.2.0"
    assert manifest["backend"] == {"entrypoint": "agenticfabriq.mnemiq.entrypoint"}
    with zipfile.ZipFile(out / "mnemiq-0.2.0.supx") as bundle:
        assert json.loads(bundle.read("manifest.json")) == manifest
    parts = manifest["backend"]["entrypoint"].split(".")
    assert (out / "dist" / "backend" / "src" / Path(*parts)).with_suffix(".py").is_file()
