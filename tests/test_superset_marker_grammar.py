"""The build marker's grammar, run through both of its parsers: mnemiq's (`mnemiq.superset.marker`) and the Superset
extension's (`agenticfabriq.mnemiq.params`). The extension runs inside Superset and cannot import mnemiq, so each
carries its own; this one table is what keeps them from drifting apart.

A marker is the last line of an object's description, `mnemiq:<build_id>:<step_key>:<dashboard_id>`: a lowercase
UUID, a step key of 1 to 40 of `[a-z0-9_]`, and a positive integer or `new` (the dashboard shell's own, made before its
id exists). Anything else, anywhere else, is no marker.
"""

import pytest

from tests.superset_extension_path import use_extension_source

use_extension_source()

from agenticfabriq.mnemiq import params as extension  # noqa: E402

from mnemiq.superset import marker as mnemiq_side  # noqa: E402

PARSERS = {"mnemiq": mnemiq_side, "extension": extension}

BUILD = "0f8e3c2a-1b4d-4c6e-9a7b-5d3e2f1a0b9c"
OTHER = "7d1e9b40-5a2c-4f3e-8b6d-0c9a8e7f6d5b"

# (description, the marker it holds as (build_id, step_key, dashboard_id), or None)
CASES = [
    ("missing", None, None),
    ("empty", "", None),
    ("all whitespace", "  \n\t \r\n ", None),
    ("a marker", f"mnemiq:{BUILD}:revenue_by_month:12", (BUILD, "revenue_by_month", 12)),
    ("dashboard 123", f"mnemiq:{BUILD}:revenue_by_month:123", (BUILD, "revenue_by_month", 123)),
    ("after prose", f"Made by mnemiq for a dashboard.\nmnemiq:{BUILD}:orders:12", (BUILD, "orders", 12)),
    ("pasted on a middle line", f"Copied from another chart:\nmnemiq:{BUILD}:orders:12\nthen edited by hand", None),
    ("prose on the marker's line", f"see mnemiq:{BUILD}:orders:12", None),
    ("marker then prose on its line", f"mnemiq:{BUILD}:orders:12 (old)", None),
    ("a step key with a space", f"mnemiq:{BUILD}:orders by month:12", None),
    ("a step key with a colon", f"mnemiq:{BUILD}:orders:month:12", None),
    ("an empty step key", f"mnemiq:{BUILD}::12", None),
    ("a 40-character step key", f"mnemiq:{BUILD}:{'k' * 40}:12", (BUILD, "k" * 40, 12)),
    ("a 41-character step key", f"mnemiq:{BUILD}:{'k' * 41}:12", None),
    ("an upper-case step key", f"mnemiq:{BUILD}:Orders:12", None),
    ("new", f"mnemiq:{BUILD}:dashboard:new", (BUILD, "dashboard", None)),
    ("NEW", f"mnemiq:{BUILD}:dashboard:NEW", None),
    ("trailing spaces", f"mnemiq:{BUILD}:orders:12   ", (BUILD, "orders", 12)),
    ("trailing newlines", f"prose\nmnemiq:{BUILD}:orders:12\n\n  \n", (BUILD, "orders", 12)),
    ("CRLF lines", f"prose\r\nmnemiq:{BUILD}:orders:12\r\n", (BUILD, "orders", 12)),
    ("two marker lines", f"mnemiq:{OTHER}:first:7\nmnemiq:{BUILD}:second:8", (BUILD, "second", 8)),
    ("a 36-hyphen build id", f"mnemiq:{'-' * 36}:orders:12", None),
    ("an upper-case build id", f"mnemiq:{BUILD.upper()}:orders:12", None),
    ("a build id without hyphens", f"mnemiq:{BUILD.replace('-', '')}:orders:12", None),
    ("dashboard 0", f"mnemiq:{BUILD}:orders:0", None),
    ("a leading zero", f"mnemiq:{BUILD}:orders:012", None),
    ("a negative dashboard id", f"mnemiq:{BUILD}:orders:-12", None),
    ("non-ASCII digits", f"mnemiq:{BUILD}:orders:１２", None),
    ("an indented marker", f"  mnemiq:{BUILD}:orders:12", None),
    ("another prefix", f"Mnemiq:{BUILD}:orders:12", None),
    ("a dashboard id too long to be an integer", f"mnemiq:{BUILD}:orders:{'9' * 5000}", None),
]


def _fields(found):
    return None if found is None else (found.build_id, found.step_key, found.dashboard_id)


@pytest.mark.parametrize("side", sorted(PARSERS))
@pytest.mark.parametrize(("name", "description", "expected"), CASES, ids=[case[0] for case in CASES])
def test_marker_grammar(side, name, description, expected):
    assert _fields(PARSERS[side].parse_marker(description)) == expected


@pytest.mark.parametrize("side", sorted(PARSERS))
def test_markers_compare_field_by_field(side):
    """Dashboard 12's marker is not dashboard 123's, though one is a prefix of the other as text."""
    parse = PARSERS[side].parse_marker
    twelve, hundred = parse(f"mnemiq:{BUILD}:orders:12"), parse(f"mnemiq:{BUILD}:orders:123")
    assert twelve != hundred and twelve.dashboard_id == 12 and hundred.dashboard_id == 123
    assert twelve == parse(f"prose\nmnemiq:{BUILD}:orders:12\n")


@pytest.mark.parametrize("side", sorted(PARSERS))
@pytest.mark.parametrize("fields", [(BUILD, "orders", 12), (OTHER, "k" * 40, 2147483647), (BUILD, "dashboard", None),
                                    (BUILD, "a_1", 1)])
def test_format_then_parse_round_trips(side, fields):
    module = PARSERS[side]
    line = module.format_marker(*fields)
    assert _fields(module.parse_marker(line)) == fields
    assert _fields(module.parse_marker(f"Some prose.\n{line}")) == fields


@pytest.mark.parametrize("side", sorted(PARSERS))
@pytest.mark.parametrize("fields", [(BUILD.upper(), "orders", 12), ("-" * 36, "orders", 12), (BUILD, "Orders", 12),
                                    (BUILD, "a b", 12), (BUILD, "", 12), (BUILD, "k" * 41, 12), (BUILD, "orders", 0),
                                    (BUILD, "orders", -1), (BUILD, "orders", True), (BUILD, "orders", "12"),
                                    (BUILD, "orders", "new"), (BUILD, "orders", 1.0)])
def test_format_refuses_what_would_not_parse_back(side, fields):
    with pytest.raises(ValueError):
        PARSERS[side].format_marker(*fields)


def test_both_sides_format_the_same_line():
    for fields in [(BUILD, "orders", 12), (BUILD, "dashboard", None)]:
        assert mnemiq_side.format_marker(*fields) == extension.format_marker(*fields)
    assert mnemiq_side.MARKER_PATTERN.pattern == extension.MARKER_PATTERN.pattern
