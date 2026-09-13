"""`reason`, the one-line exception summary every warehouse script uses.

It exists because `str(exc).splitlines()[-1]` -- the idiom it replaces -- raises IndexError on
an empty message, and every one of the eight sites using it sat inside an `except` whose whole
purpose was to keep a failure from ending the run. The IndexError escaped the handler and did
exactly what the handler existed to prevent.

Shared rather than fixed in place because correcting one site is how it survived: a commit
message claimed the idiom had been routed through a local helper everywhere, and one of the
three uses in that same file had been left behind.
"""

import pytest

from scripts.exc_reason import reason


def test_an_empty_message_yields_the_class_name_instead_of_raising():
    """`"".splitlines()` is `[]`, so the old idiom's `[-1]` raised here."""
    assert reason(RuntimeError("")) == "RuntimeError"


def test_the_LAST_line_is_the_useful_one():
    """Snowflake stacks context above the actual error."""
    assert reason(RuntimeError("context\nmore context\n001003 (42000): SQL error")) == (
        "001003 (42000): SQL error"
    )


def test_a_single_line_survives_whole():
    assert reason(ValueError("just this")) == "just this"


def test_surrounding_whitespace_is_trimmed_from_the_line_it_returns():
    """Snowflake indents its continuation lines. Without the strip the summary carries the
    indentation into a column-aligned report, and no other case here has any -- mutating the
    return to drop `.strip()` left the whole file green."""
    assert reason(RuntimeError("ctx\n   001003 (42000): SQL error   ")) == (
        "001003 (42000): SQL error"
    )


def test_the_limit_truncates_and_is_per_caller():
    assert reason(RuntimeError("x" * 200)) == "x" * 70
    assert reason(RuntimeError("x" * 200), 30) == "x" * 30


@pytest.mark.parametrize("exc", [RuntimeError(""), RuntimeError("\n"), ValueError()])
def test_no_input_makes_it_raise(exc):
    """The property that matters: this is called from handlers that must not fail."""
    assert reason(exc), f"empty summary for {exc!r}"
