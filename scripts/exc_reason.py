"""One short line from an exception, safely.

`str(exc).splitlines()[-1]` is the idiom every script here used to summarise a Snowflake
error, and it raises IndexError when the message is empty -- `"".splitlines()` is `[]`. Every
one of those uses sits inside an `except` whose whole purpose is to keep a failure from ending
the run, so the IndexError would escape the handler and do precisely what the handler exists
to prevent.

Shared rather than fixed in place, because it was in eight sites across six scripts --
enumerated, not counted -- and correcting one of them is how it survived the first time.
"""

from __future__ import annotations


def reason(exc: Exception, limit: int = 70) -> str:
    r"""Last non-blank line of the message, or the exception's class name when there is none.

    NON-BLANK, not merely present: `"\n".splitlines()` is `[""]`, so guarding only on the list
    being empty returned an empty summary and the caller printed `(probe failed: )`. Not a
    crash, but a handler that reports nothing is the same problem one step quieter.
    """
    for line in reversed(str(exc).splitlines()):
        if line.strip():
            return line.strip()[:limit]
    return exc.__class__.__name__[:limit]
