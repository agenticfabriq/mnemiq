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


def _packer():
    spec = importlib.util.spec_from_file_location("build_superset_extension",
                                                  ROOT / "scripts" / "build_superset_extension.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(tmp_path):
    out = tmp_path / "out"
    assert _packer().main(["--out", str(out)]) == 0
    return out


def test_bundle_has_only_allowed_entries(tmp_path):
    out = _build(tmp_path)
    with zipfile.ZipFile(out / "mnemiq-0.1.0.supx") as bundle:
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
    assert manifest["id"] == "agenticfabriq.mnemiq" and manifest["version"] == "0.1.0"
    assert manifest["backend"] == {"entrypoint": "agenticfabriq.mnemiq.entrypoint"}
    with zipfile.ZipFile(out / "mnemiq-0.1.0.supx") as bundle:
        assert json.loads(bundle.read("manifest.json")) == manifest
    parts = manifest["backend"]["entrypoint"].split(".")
    assert (out / "dist" / "backend" / "src" / Path(*parts)).with_suffix(".py").is_file()
