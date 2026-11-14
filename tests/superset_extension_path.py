"""Puts the Superset extension's backend source on `sys.path`, so a test imports its pure modules (`token`,
`bundle` helpers) without installing the extension or importing Superset or Flask."""

from __future__ import annotations

import sys
from pathlib import Path

EXTENSION_ROOT = Path(__file__).resolve().parent.parent / "integrations" / "superset"
BACKEND_SRC = EXTENSION_ROOT / "backend" / "src"


def use_extension_source() -> None:
    if str(BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(BACKEND_SRC))
