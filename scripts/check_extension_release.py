"""Checks a release tag against the Superset extension's manifest, and names the bundle the release carries.

A tag `superset-extension-v<version>` must name exactly the version `integrations/superset/extension.json` holds, so
the bundle a release publishes is the one its tag says. Standard library only, as the extension's packer is.

    python scripts/check_extension_release.py superset-extension-v0.2.0     # prints mnemiq-0.2.0.supx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PREFIX = "superset-extension-v"
MANIFEST = Path(__file__).resolve().parent.parent / "integrations" / "superset" / "extension.json"


def release_bundle(tag: str, manifest: Path = MANIFEST) -> str:
    """The bundle file a release tagged `tag` carries, `<name>-<version>.supx`; ValueError when the tag does not name
    the manifest's version."""
    found = json.loads(manifest.read_text())
    version, name = found["version"], found["name"]
    if tag != f"{PREFIX}{version}":
        raise ValueError(f"the tag {tag!r} does not name the extension's version: tag {PREFIX}{version} for {version}, "
                         f"or change the version in {manifest.name}")
    return f"{name}-{version}.supx"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_extension_release.py <tag>", file=sys.stderr)
        return 2
    try:
        print(release_bundle(argv[0]))
    except ValueError as refused:
        print(refused, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
