"""Builds the mnemiq Superset extension: `DIR/dist/` and `DIR/<name>-<version>.supx`.

Standard library only. The bundle holds exactly the entries Superset 7.0.0rc2's loader accepts: `manifest.json`,
`frontend/dist/...` and `backend/src/...` (`superset/extensions/utils.py`, `FRONTEND_REGEX` and `BACKEND_REGEX`).
This extension has no frontend yet, so it writes `manifest.json` and the backend sources only.

    python scripts/build_superset_extension.py --out build/superset-extension
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

EXTENSION = Path(__file__).resolve().parent.parent / "integrations" / "superset"


def _backend_files(source: Path) -> list[Path]:
    """The backend's `.py` files under `src/`, in a fixed order, never a cache."""
    return sorted(p for p in (source / "backend" / "src").rglob("*.py") if "__pycache__" not in p.parts)


def build(out: Path, extension: Path = EXTENSION) -> Path:
    config = json.loads((extension / "extension.json").read_text())
    publisher, name = config["publisher"], config["name"]
    entrypoint = f"{publisher.replace('-', '_')}.{name.replace('-', '_')}.entrypoint"
    manifest = {"id": f"{publisher}.{name}", "publisher": publisher, "name": name,
                "displayName": config.get("displayName", name), "version": config["version"],
                "license": config.get("license", "Apache-2.0"), "backend": {"entrypoint": entrypoint}}
    dist = out / "dist"
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    (dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    entries = ["manifest.json"]
    for path in _backend_files(extension):
        entry = "backend/src/" + path.relative_to(extension / "backend" / "src").as_posix()
        (dist / entry).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dist / entry)
        entries.append(entry)
    bundle = out / f"{name}-{config['version']}.supx"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zipped:
        for entry in entries:
            zipped.write(dist / entry, entry)
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="the directory to write dist/ and the .supx into")
    bundle = build(parser.parse_args(argv).out)
    print(f"wrote {bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
