"""The words each engine reserves: never read as a name where they stand unquoted, not after a dot.

A row filter runs as written, and its name count (`rls._unaccounted_names`) refuses a word the
parse loses where a name could stand, since the engine may read a column there. A word the
engine reserves it reads as a keyword wherever it stands, unless after a dot, so losing it loses
no name: `TRIM(BOTH FROM region)` and `SELECT ALL id` were refused for BOTH and ALL (Codex's
seventh review of #193). After a dot Postgres and DuckDB read a reserved word as a column label
(`e.all`), so a word after a dot is not taken as reserved.

Measured on each engine, the lists below are subsets of what it reports. An engine with no list
here (SQLite, which takes most of its keywords as names where a name fits) excuses nothing. Each
list is checked against the engine: DuckDB's against the installed library, Oracle's and
Postgres's live where a test database is set.
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

RESERVED: dict[str, frozenset[str]] = {
    "duckdb": frozenset(_DUCKDB.split()),
    "postgres": frozenset(_POSTGRES.split()),
    "oracle": frozenset(_ORACLE.split()),
}


def reserved(word: str, dialect: str | None) -> bool:
    """Whether `dialect`'s engine reserves `word`, compared casefolded."""
    return word.casefold() in RESERVED.get((dialect or "").lower(), frozenset())


# Words each engine will not read as a bare TABLE or alias name, so the row-filter rewrite's
# derived table must quote them there (M233). This is BROADER than `reserved` above and a
# different question: `reserved` is "a keyword wherever it stands" for the filter name-count,
# while a table position also rejects the join and operator keywords (`inner`, `join`, `like`,
# `is`) that a filter reads as a column label. Built from the engine's own parser: a word is
# listed where `SELECT 1 FROM <word>` is a syntax error (parseable -> "no such table" is not).
#
# DuckDB and SQLite are probed exhaustively and complete. Postgres and Oracle reuse their
# `reserved` lists above as a sound subset, not proven complete: a served postgres or sqlite
# source runs through DuckDB (ATTACH), so DuckDB's list governs it, and only a native Oracle
# source runs as oracle. An engine keyword outside its list here renders bare and the engine
# then refuses the governed query -- fail-closed availability, never a disclosure. (M233 residual.)
_DUCKDB_TABLE = """
all analyse analyze and anti any array as asc asof asymmetric at authorization binary both by case
cast check collate collation column concurrently constraint create cross default deferrable desc
describe distinct do else end except false fetch for foreign freeze from full glob group having
ilike in initially inner intersect into is isnull join lambda lateral leading left like limit
natural not notnull null offset on only or order outer overlaps pivot pivot_longer pivot_wider
placing positional primary qualify references returning right select semi show similar some
summarize symmetric table tablesample then to trailing true union unique unpack unpivot using
variadic verbose when where window with
"""  # every duckdb_keywords() word that `SELECT 1 FROM <word>` rejects, DuckDB 1.5.4

_SQLITE_TABLE = """
add all alter and as autoincrement between case check collate commit constraint create default
deferrable delete distinct drop else escape except exists foreign from group having in index insert
intersect into is isnull join limit not nothing notnull null on or order primary references
returning select set table then to transaction union unique update using values when where
"""  # every sqlite3_keyword_name() word that `SELECT 1 FROM <word>` rejects, SQLite 3.50.4

BARE_TABLE_UNSAFE: dict[str, frozenset[str]] = {
    "duckdb": frozenset(_DUCKDB_TABLE.split()),
    "sqlite": frozenset(_SQLITE_TABLE.split()),
    "postgres": frozenset(_POSTGRES.split()),
    "oracle": frozenset(_ORACLE.split()),
}


def bare_table_reserved(word: str, dialect: str | None) -> bool:
    """Whether `dialect` refuses `word` as a bare table or alias name, so the row-filter
    rewrite's derived table must quote it there (M233). Casefolded. See BARE_TABLE_UNSAFE."""
    return word.casefold() in BARE_TABLE_UNSAFE.get((dialect or "").lower(), frozenset())
