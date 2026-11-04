"""The words each engine reserves: never read as a name where they stand unquoted, not after a dot.

A row filter runs as written, and its name count (`rls._unaccounted_names`) refuses a word the
parse loses where a name could stand, since the engine may read a column there. A word the
engine reserves it reads as a keyword wherever it stands, unless after a dot, so losing it loses
no name: `TRIM(BOTH FROM region)` and `SELECT ALL id` were refused for BOTH and ALL (Codex's
seventh review of #193). After a dot Postgres and DuckDB read a reserved word as a column label
(`e.all`), so a word after a dot is not taken as reserved.

Measured on each engine, the lists below are subsets of what it reports. A dialect with no list
here excuses nothing. Each list is checked against the engine: DuckDB's against the installed
library, Oracle's and Postgres's live where a test database is set, and SQLite's against the
bundled library -- SQLite takes most of its 147 keywords as names where a name fits, so its list
is only the 58 its parser rejects as a bare table name (`null` and `select` among them, `true`
and `false` not), which otherwise break the governed derived table a row filter builds (M233).
"""
from __future__ import annotations

_DUCKDB = """
all analyse analyze and any array as asc asymmetric both case cast check collate column constraint
create default deferrable desc describe distinct do else end except false fetch for foreign from
group having in initially intersect into lambda lateral leading limit not null offset on only or
order pivot pivot_longer pivot_wider placing primary qualify references returning select show some
summarize symmetric table then to trailing true union unique unpivot using variadic when where
window with
"""  # duckdb_keywords() WHERE keyword_category = 'reserved', DuckDB 1.5.4

_POSTGRES = """
all analyse analyze and any array as asc asymmetric both case cast check collate column constraint
create current_catalog current_date current_role current_time current_timestamp current_user
default deferrable desc distinct do else end except false fetch for foreign from grant group having
in initially intersect into lateral leading limit localtime localtimestamp not null offset on only
or order placing primary references returning select session_user some symmetric table then to
trailing true union unique user using variadic when where window with
"""  # pg_get_keywords() WHERE catcode = 'R', PostgreSQL 15.18

_ORACLE = """
all alter and any as asc between by char check cluster compress connect create date decimal default
delete desc distinct drop else except exclusive exists float for from grant group having identified
in index insert integer intersect into is like lock long minus mode nocompress not nowait null number
of on option or order pctfree prior public raw rename resource revoke select set share size smallint
start synonym table then to trigger union unique update values varchar varchar2 view where with
"""  # V$RESERVED_WORDS WHERE reserved = 'Y', its words, Oracle Free 23

_SQLITE = """
add all alter and as autoincrement between case check collate commit constraint create default
deferrable delete distinct drop else escape except exists foreign from group having in index insert
intersect into is isnull join limit not nothing notnull null on or order primary references
returning select set table then to transaction union unique update using values when where
"""  # sqlite3_keyword_name(), kept where `SELECT 1 FROM <kw>` is a syntax error (bare table), SQLite 3.50.4

RESERVED: dict[str, frozenset[str]] = {
    "duckdb": frozenset(_DUCKDB.split()),
    "postgres": frozenset(_POSTGRES.split()),
    "oracle": frozenset(_ORACLE.split()),
    "sqlite": frozenset(_SQLITE.split()),
}


def reserved(word: str, dialect: str | None) -> bool:
    """Whether `dialect`'s engine reserves `word`, compared casefolded."""
    return word.casefold() in RESERVED.get((dialect or "").lower(), frozenset())
