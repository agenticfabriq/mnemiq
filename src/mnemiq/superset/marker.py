"""The marker a dashboard build leaves on every object it creates, so a step whose outcome is uncertain can find its
object again, and no build touches another's.

A marker is the last line of the object's description, after any prose: `mnemiq:<build_id>:<step_key>:<dashboard_id>`.
The build id is a lowercase UUID, the step key 1 to 40 of `[a-z0-9_]`, and the dashboard id a positive integer, or
`new` on the dashboard shell itself, whose id does not exist until it is created. Markers are compared field by field,
never as text, so dashboard 12's is not dashboard 123's.

The Superset extension carries the same parser (`integrations/superset/backend/src/agenticfabriq/mnemiq/params.py`),
since it runs inside Superset and cannot import mnemiq. `tests/test_superset_marker_grammar.py` runs one table of cases
through both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["MARKER_PATTERN", "Marker", "format_marker", "parse_marker"]

MARKER_PATTERN = re.compile(
    r"mnemiq:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}):([a-z0-9_]{1,40}):([1-9][0-9]*|new)")


@dataclass(frozen=True)
class Marker:
    """A build's marker: the build, the step, and the dashboard (None: `new`, the dashboard shell's own)."""

    build_id: str
    step_key: str
    dashboard_id: int | None


def _is_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def parse_marker(description: str | None) -> Marker | None:
    """The marker on the last line of `description`, or None: an empty or missing description, a last line that is
    not exactly a marker, and a marker on any other line are all no marker."""
    lines = (description or "").rstrip().splitlines()
    if not lines:
        return None
    found = MARKER_PATTERN.fullmatch(lines[-1])
    if found is None:
        return None
    build_id, step_key, dashboard = found.groups()
    if dashboard == "new":
        return Marker(build_id, step_key, None)
    try:
        return Marker(build_id, step_key, int(dashboard))
    except ValueError:  # more digits than Python converts to an integer: no dashboard has that id
        return None


def format_marker(build_id: str, step_key: str, dashboard_id: int | None) -> str:
    """The marker line for these fields (`dashboard_id` None writes `new`). Refuses fields that would not parse back
    to themselves."""
    if dashboard_id is None:
        tail = "new"
    elif _is_id(dashboard_id):
        tail = str(dashboard_id)
    else:
        raise ValueError("a marker's dashboard id is a positive integer, or None for new")
    if not isinstance(build_id, str) or not isinstance(step_key, str):
        raise ValueError("a marker's build id and step key are strings")
    line = f"mnemiq:{build_id}:{step_key}:{tail}"
    if parse_marker(line) != Marker(build_id, step_key, dashboard_id):
        raise ValueError("these fields do not make a marker")
    return line
