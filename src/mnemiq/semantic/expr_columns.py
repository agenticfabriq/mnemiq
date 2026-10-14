"""The columns a dimension's expression reads -- one parser, shared.

Moved from `enrichment.certified` (M136): `apply_certified` stamps each column a personal dimension
reads with the dimension's level (M135), and `select_dimensions` must decide which dimensions to
offer from the SAME reading of the expression, or the two disagree about which column a dimension
reads and the packet offers a grouping the decider refuses.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp


# The dialects a dimension's `expr` is read in. It carries none of its own, so it is read as each
# would: the default (double quotes) and sqlite, which takes backticks and brackets as well -- the
# quoting MySQL, BigQuery and SQL Server write. A column any of them reads is read.
_EXPR_DIALECTS: tuple[str | None, ...] = (None, "sqlite")
_QUOTES = '"`[]'

# What reads as itself, never as a same-named column, on DuckDB, Postgres, sqlite and MySQL
# alike (each measured with the column on the table): a plain integer, NULL. Matched on the
# spelling, not on sqlglot's parse -- sqlglot reads `1e` as a number and MySQL reads it as a
# column named `1e` (Codex's review of #100). Anything else takes the spelling, which requires
# more clearance, never less: `1e5`, `-42`, and `true` and `current_date`, which sqlite and DuckDB
# read as a column so named (Codex's review of #96). ASCII digits: `\d` matches every script's.
_CONSTANT = re.compile(r"[0-9]+|null", re.IGNORECASE)


def _names_table(qualifier: str, table: str) -> bool:
    """Whether a column's qualifier names `table`, compared on dotted suffixes: `pg.person` and
    `person` name each other. A same-named table of another schema matches too, which requires
    more clearances, never fewer."""
    q, t = qualifier.lower(), table.lower()
    return not q or q == t or t.endswith("." + q) or q.endswith("." + t)


def columns_read(expr: str, source: str) -> set[frozenset[str]]:
    """The columns of `source` that `expr` reads, each as the folded names it could be.

    Every column it reads, not only an `expr` that IS one: a personal value derived from columns
    -- `lower(ssn)` -- makes them personal, and reading them requires its clearance as reading it
    would. A column the expression only tests (`status` in `CASE WHEN status = 'x' THEN ssn
    END`) takes the level too, deliberately: it can deny a column that is not itself personal to
    the uncleared, which refuses more, and narrowing it would mean deciding which reads expose.

    Parsed by sqlglot in each of `_EXPR_DIALECTS`. A name read with a dot in it is a quoted path
    read as one identifier -- BigQuery's `` `customer.ssn` `` is one name to sqlite (Codex's review
    of the option-A change, which measured it leaving `ssn` open) -- so it is also split: the last
    part the column, the rest its qualifier. Both readings are kept, the split one and the whole
    one a column may truly be named, and a reference is one set of the names it could be.

    Where no dialect reads a column, the spelling read: the last dotted part, if what precedes it
    -- its quotes off -- names the table. sqlglot reads an unquoted `comment` as a command and
    `customer.select` as nothing, and the gate measured each leaving a keyword-named column open.
    Not where the expression is a constant (`_CONSTANT`): `42` reads no column named `42`.
    """
    refs: set[frozenset[str]] = set()
    for dialect in _EXPR_DIALECTS:
        try:
            node = sqlglot.parse_one(expr, read=dialect)
        except sqlglot.errors.SqlglotError:
            continue
        if node is None:
            continue
        for column in node.find_all(exp.Column):
            if not column.name:
                continue
            qualifier = ".".join(p for p in (column.catalog, column.db, column.table) if p)
            names = set()
            if _names_table(qualifier, source):
                names.add(column.name.lower())
            path, _, last = column.name.rpartition(".")
            if path and last and _names_table(".".join(p for p in (qualifier, path) if p), source):
                names.add(last.lower())
            if names:
                refs.add(frozenset(names))
    if refs:
        return refs
    if _CONSTANT.fullmatch(expr.strip()):
        return set()
    qualifier, _, name = expr.strip().rpartition(".")
    qualifier = ".".join(part.strip(_QUOTES) for part in qualifier.split(".")) if qualifier else ""
    return {frozenset({name.lower()})} if name and _names_table(qualifier, source) else set()


def expression_scope(expr: str, source: str) -> str:
    """Whether `columns_read` can answer for `expr` over `source`: `"own"`, `"foreign"` or
    `"unparsed"` (M144, Codex's review of #121).

    `columns_read` names the columns of SOURCE an expression reads, so an expression that reads
    anything else -- a subquery, or a column another table qualifies -- reads columns it never
    reports, and a check built on it passed `(SELECT count(p.secret_code) FROM patients p)` over
    `orders`. That is `"foreign"`. `"unparsed"` is an expression no dialect parses, for which
    `columns_read` falls back to its spelling; the caller decides whether that spelling names a
    real column (a keyword-named one does) or nothing (a malformed expression does).
    """
    for dialect in _EXPR_DIALECTS:
        try:
            node = sqlglot.parse_one(expr, read=dialect)
        except sqlglot.errors.SqlglotError:
            continue
        if node is None:
            continue
        if node.find(exp.Query) is not None:
            return "foreign"
        for column in node.find_all(exp.Column):
            qualifier = ".".join(p for p in (column.catalog, column.db, column.table) if p)
            if qualifier and not _names_table(qualifier, source):
                return "foreign"
        return "own"
    return "unparsed"
