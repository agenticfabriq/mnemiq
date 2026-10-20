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
        aliases = _source_aliases(node, source)
        for column in node.find_all(exp.Column):
            if not column.name:
                continue
            qualifier = ".".join(p for p in (column.catalog, column.db, column.table) if p)
            names = set()
            if _names_table(qualifier, source) or qualifier.lower() in aliases:
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


def _table_name(table: exp.Table) -> str:
    return ".".join(p for p in (table.catalog, table.db, table.name) if p)


def _is_source(table: exp.Table, source: str) -> bool:
    """Whether a RELATION is the source table: the same dotted name, or a bare name equal to the
    source's own last part. Not `_names_table`, whose suffix match serves column qualifiers,
    where matching more tables demands more clearance: as a relation test it admitted
    `private.orders` as `orders` (Codex's review of #172), another schema's table read as the
    source and its columns checked against the source's grant. One direction only, on
    purpose: a bare source claims the unqualified table, so a qualified relation over it
    (`pg.orders` over `orders`) may be another schema's and is foreign, the metric withheld
    (the review gate on the second commit); a bundle that qualifies its tables registers its
    sources qualified. A quoted name keeps its case and must match as written (`"ORDERS"` is
    not `orders` where quoting preserves case: the gate's second advisory)."""
    quoted = bool(table.this.args.get("quoted")) if isinstance(table.this, exp.Identifier) else False
    name = _table_name(table)
    t, s = (name, source) if quoted else (name.lower(), source.lower())
    return t == s or ("." not in t and t == s.rpartition(".")[2])


def _source_aliases(node: exp.Expression, source: str) -> set[str]:
    """The aliases the expression's own subqueries give `source` (`FROM orders r`): a column `r`
    qualifies is a column of the source."""
    return {table.alias.lower() for table in node.find_all(exp.Table)
            if table.alias and _is_source(table, source)}


# A relation `columns_read` does not follow: a CTE, a LATERAL, a VALUES list, an UNNEST. A
# subquery over the source itself it does follow, so only these make an expression foreign.
_FOREIGN_RELATIONS = (exp.With, exp.Lateral, exp.Values, exp.Unnest)


def expression_scope(expr: str, source: str) -> str:
    """Whether `columns_read` can answer for `expr` over `source`: `"own"`, `"foreign"` or
    `"unparsed"` (M144, Codex's review of #121).

    `columns_read` names the columns of SOURCE an expression reads, so an expression that reads
    anything else -- another table, in a subquery or qualifying a column -- reads columns it
    never reports, and a check built on it passed `(SELECT count(p.secret_code) FROM patients
    p)` over `orders`. That is `"foreign"`. A subquery over the SOURCE ALONE is `"own"`: every
    column it reads, aliased (`FROM orders r ... r.stamp`), unqualified or through a derived
    table, is a column of the source, which `columns_read` reports, so the raw-read check still
    covers each. Read as foreign for any subquery (the first gate), a certified metric carrying
    the latest-version filter `x = (SELECT max(r.x) FROM own_table r WHERE r.id = own_table.id)`
    was withheld from every identity, and four of a pilot bundle's five metrics never reached
    the model (the ui lane's measurement, 2026-10-20). `"unparsed"` is an expression no dialect
    parses, for which `columns_read` falls back to its spelling; the caller decides whether that
    spelling names a real column (a keyword-named one does) or nothing (a malformed expression
    does).
    """
    for dialect in _EXPR_DIALECTS:
        try:
            node = sqlglot.parse_one(expr, read=dialect)
        except sqlglot.errors.SqlglotError:
            continue
        if node is None:
            continue
        if node.find(*_FOREIGN_RELATIONS) is not None:
            return "foreign"
        if node.find(exp.Columns) is not None or any(
                not isinstance(star.parent, exp.Count) for star in node.find_all(exp.Star)):
            # A star reads columns no node names, so `columns_read` cannot report them: `(SELECT
            # max(q.secret_code) FROM (SELECT * FROM orders) q)` reported nothing and passed the
            # uncleared (the review gate on the first draft). COUNT(*) alone reads none.
            return "foreign"
        for table in node.find_all(exp.Table):
            if not _table_name(table) or not _is_source(table, source):
                return "foreign"  # another table, or a table function, which has no name here
        if any(join.args.get("using") or (join.args.get("method") or "").upper() == "NATURAL"
               for join in node.find_all(exp.Join)):
            # A USING key is no column node and a NATURAL join names nothing, so a self-join's
            # keys are read by nothing: `(SELECT count(*) FROM orders o JOIN orders r
            # USING (secret_code))` was offered to the uncleared (Codex's review of #172).
            return "foreign"
        if any(alias.args.get("columns") for alias in node.find_all(exp.TableAlias)):
            # An alias column list renames by position (`FROM orders q(a, c)` makes `q.c` the
            # second column, whichever that is), which `columns_read` reports under the new
            # name: the review gate on the second draft, the withheld column read as `c`.
            return "foreign"
        source_aliases = _source_aliases(node, source)
        aliases = source_aliases | {
            derived.alias.lower() for derived in node.find_all(exp.Subquery) if derived.alias}
        whole_row = source_aliases | {source.lower(), source.lower().rpartition(".")[2]}
        for column in node.find_all(exp.Column):
            qualifier = ".".join(p for p in (column.catalog, column.db, column.table) if p)
            if qualifier and not _names_table(qualifier, source) and qualifier.lower() not in aliases:
                return "foreign"
            if not qualifier and column.name.lower() in whole_row:
                # The whole row as a value (`to_json(r)`, `struct_pack(orders)`): every column,
                # reported as none (the review gate on the first draft).
                return "foreign"
        return "own"
    return "unparsed"
