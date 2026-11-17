"""A date part written as a function's unit the way another SQL dialect writes it, named as a unit.

`TIMESTAMPDIFF(HOUR, s, e)` is MySQL's spelling. Read as DuckDB or Postgres, `HOUR` is a bare identifier, so the
access check refuses it as a column no table has: true of the tree it read, and no help to a writer that never meant
a column, which then sends the same statement again. Under `repair_hints` the refusal says what the word is, and how
the writer's own dialect measures the difference between two times.

The unit's position is read from the statement as written, not from sqlglot's tree. sqlglot moves a foreign
function's arguments (DuckDB reads `TIMESTAMPDIFF(HOUR, s, e)` as `DATE_DIFF('E', s, HOUR)`), and asking only whether
the column sits inside some function would call `hour` in `SUM(hour)` a unit. Only the first argument of a named
date function is a unit.

No rewritten call is offered. sqlglot's transpile of these functions is wrong for several targets (MySQL's
`TIMESTAMPDIFF` to Postgres comes out as `TIMESTAMPDIFF(e, s, HOUR)`), and the same unit means different things
across dialects: DuckDB's `date_diff` counts boundaries crossed where MySQL's `TIMESTAMPDIFF` counts whole units. A
sentence per dialect, each claim executed by the tests, cannot hand the writer a wrong call to copy.
"""

from __future__ import annotations

import re
from dataclasses import replace

from mnemiq.sql.verdict import Refusal, RefusalCode

DATE_PARTS = frozenset({"microsecond", "millisecond", "second", "minute", "hour", "day", "week", "month", "quarter",
                        "year"})
DIFFERENCE_FUNCTIONS = frozenset({"DATEDIFF", "DATE_DIFF", "TIMESTAMPDIFF", "TIMESTAMP_DIFF"})
UNIT_FUNCTIONS = DIFFERENCE_FUNCTIONS | frozenset({"DATEADD", "DATE_ADD", "TIMESTAMPADD", "TIMESTAMP_ADD", "DATEPART",
                                                   "DATE_PART", "DATENAME", "DATE_TRUNC", "DATETRUNC",
                                                   "TIMESTAMP_TRUNC"})

_OWN_FUNCTIONS = "Write it with this database's own date functions."
# How each dialect measures the time between two values: the claim each makes is executed in the tests.
_DIFFERENCE: dict[str, str] = {
    "duckdb": ("In duckdb, date_sub('{part}', start, end) counts whole {part}s elapsed and date_diff('{part}', start, "
               "end) counts {part} boundaries crossed, with the unit as a quoted string."),
    "postgres": ("In Postgres, EXTRACT(EPOCH FROM end - start) is the seconds between two timestamps, to divide down "
                 "to the unit wanted."),
    "sqlite": ("In SQLite, julianday(end) - julianday(start) is the days between two times, to multiply up to the "
               "unit wanted."),
    "oracle": ("In Oracle, subtracting two DATEs gives the days between them as a number, and subtracting two "
               "TIMESTAMPs gives an INTERVAL DAY TO SECOND."),
}
_DIFFERENCE["postgresql"] = _DIFFERENCE["postgres"]


def _unit_call(sql: str, name: str) -> str | None:
    """The date function, upper case, that `sql` calls with `name` as its first argument; None when it calls none."""
    found = re.search(r"\b(" + "|".join(sorted(UNIT_FUNCTIONS, key=len, reverse=True)) + r")\s*\(\s*"
                      + re.escape(name) + r"\s*,", sql, flags=re.IGNORECASE)
    return found.group(1).upper() if found else None


def date_unit_refusal(sql: str, refusal: Refusal, dialect: str | None) -> Refusal | None:
    """`refusal` reworded when it refuses as a missing column a date part `sql` writes as a date function's unit; None
    for every other refusal, which then stands as it is."""
    subject = refusal.subject
    if refusal.code is not RefusalCode.UNKNOWN_COLUMN or not subject or subject.lower() not in DATE_PARTS:
        return None
    func = _unit_call(sql, subject)
    if func is None:
        return None
    part = subject.lower()
    arithmetic = (_DIFFERENCE.get((dialect or "").lower(), _OWN_FUNCTIONS).format(part=part)
                  if func in DIFFERENCE_FUNCTIONS else _OWN_FUNCTIONS)
    return replace(refusal, message=(f"{subject!r} is not a column here: it is the unit argument of {func}, written "
                                     f"the way another SQL dialect writes it. {arithmetic}"))
