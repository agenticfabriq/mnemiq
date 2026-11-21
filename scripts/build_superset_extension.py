"""Builds the mnemiq Superset extension: `DIR/dist/` and `DIR/<name>-<version>.supx`.

Standard library only. The bundle holds exactly the entries Superset 7.0.0rc2's loader accepts: `manifest.json`,
`frontend/dist/...` and `backend/src/...` (`superset/extensions/utils.py`, `FRONTEND_REGEX` and `BACKEND_REGEX`).
The frontend is the panel's built `dist/` (`integrations/superset/frontend/dist`, made by its `pnpm build`), packed
when it is there; `--require-frontend` refuses to pack without one. Superset serves every extension file `immutable`
for a year (`ExtensionsRestApi.content`), so each frontend file's name must carry a content hash, and no source map
is packed.

    python scripts/build_superset_extension.py --out build/superset-extension [--require-frontend]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

EXTENSION = Path(__file__).resolve().parent.parent / "integrations" / "superset"


def _backend_files(source: Path) -> list[Path]:
    """The backend's `.py` files under `src/`, in a fixed order, never a cache."""
    return sorted(p for p in (source / "backend" / "src").rglob("*.py") if "__pycache__" not in p.parts)


# A built file's name carries its content hash (`[name].[contenthash].js`), as every name served `immutable` must.
_HASHED = re.compile(r"\.[0-9a-f]{8,}\.")
_REMOTE_ENTRY = re.compile(r"^remoteEntry\.[0-9a-f]{8,}\.js$")


def _camel(kebab: str) -> str:
    head, *rest = kebab.split("-")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _frontend_files(dist: Path) -> tuple[list[Path], str]:
    """The frontend's files under `dist`, in a fixed order, and its one remote entry's name; refused (ValueError):
    a file whose name carries no content hash, a source map, and none or several remote entries."""
    files = sorted(p for p in dist.rglob("*") if p.is_file())
    for path in files:
        if path.suffix == ".map":
            raise ValueError(f"the frontend holds a source map, {path.name}, which is not packed")
        if not _HASHED.search(path.name):
            raise ValueError(f"the frontend file {path.name} carries no content hash in its name")
    entries = [p.relative_to(dist).as_posix() for p in files if _REMOTE_ENTRY.match(p.name) and p.parent == dist]
    if len(entries) != 1:
        raise ValueError(f"the frontend must hold one remote entry (remoteEntry.<hash>.js), not {len(entries)}")
    return files, entries[0]


def build(out: Path, extension: Path = EXTENSION, *, frontend_dist: Path | None = None,
          require_frontend: bool = False) -> Path:
    """`out/dist/` and `out/<name>-<version>.supx`, with the frontend at `frontend_dist` (else the extension's own
    `frontend/dist` when it is there). `require_frontend` refuses (ValueError) to pack without one."""
    config = json.loads((extension / "extension.json").read_text())
    publisher, name = config["publisher"], config["name"]
    entrypoint = f"{publisher.replace('-', '_')}.{name.replace('-', '_')}.entrypoint"
    manifest = {"id": f"{publisher}.{name}", "publisher": publisher, "name": name,
                "displayName": config.get("displayName", name), "version": config["version"],
                "license": config.get("license", "Apache-2.0"), "backend": {"entrypoint": entrypoint}}
    source = frontend_dist if frontend_dist is not None else extension / "frontend" / "dist"
    frontend: tuple[list[Path], str] | None = None
    if source.is_dir():
        frontend = _frontend_files(source)
        # rc2's CLI names the container `<publisher>_<name>` in camel case (`get_module_federation_name`).
        manifest["frontend"] = {"remoteEntry": frontend[1],
                                "moduleFederationName": f"{_camel(publisher)}_{_camel(name)}"}
    elif require_frontend:
        raise ValueError(f"no built frontend at {source}: run `pnpm build` in integrations/superset/frontend")
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
    for path in frontend[0] if frontend else []:
        entry = "frontend/dist/" + path.relative_to(source).as_posix()
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
    parser.add_argument("--require-frontend", action="store_true", help="refuse to pack without a built frontend")
    args = parser.parse_args(argv)
    bundle = build(args.out, require_frontend=args.require_frontend)
    print(f"wrote {bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
