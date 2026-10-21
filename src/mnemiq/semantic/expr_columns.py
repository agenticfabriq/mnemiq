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
    of the option-A change, which measured it leaving `ssn` open) -- so it is also walked prefix
    by prefix: the part after a prefix naming the source or its alias is the column, whatever
    follows (`r.secret_code.value`, a nested field path), and an unqualified path's first part is
    the column it may be as a struct (`secret_code.value`: Codex's seventh review of #172). Every
    reading is kept, the whole name a column may truly be named among them, and a reference is
    one set of the names it could be.

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
                names.add(column.name.lower())  # the whole name as read, dotted or not
            # A dotted name, qualified by the parser or read whole by a dialect, is walked
            # prefix by prefix as well: the part right after a prefix that names the source, or
            # an alias the expression gives it, is the column, whatever follows it
            # (`\`r.secret_code.value\`` over `orders r` is a nested field path to BigQuery's
            # reader, and split on its last dot it named nothing: Codex's sixth review of #172;
            # `\`r.secret_code\`` the same on its second). An unqualified path's first part is
            # the column it may be as a struct whose field follows (`\`secret_code.value\``:
            # the seventh review). Every prefix is tried, the whole reading kept beside them.
            parts = [p for p in (qualifier.split(".") if qualifier else []) + column.name.split(".") if p]
            if not qualifier and len(parts) > 1:
                names.add(parts[0].lower())
            for k in range(1, len(parts)):
                head = ".".join(parts[:k])
                if _names_table(head, source) or (k == 1 and head.lower() in aliases):
                    names.add(parts[k].lower())
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
    sources qualified. Each dotted part is compared on its own quoting: a quoted part keeps
    its case and must match as written, an unquoted one folds (`"ORDERS"` is not `orders`,
    nor is `"PG".orders` `pg.orders`, where quoting preserves case: the gate's second and
    third advisories)."""
    slots = [part for part in (table.args.get("catalog"), table.args.get("db"), table.this) if part is not None]
    if not all(isinstance(part, exp.Identifier) for part in slots):
        return False  # a four-part name puts its tail in the table slot as a Dot (the gate)
    parts = [(part.name, bool(part.args.get("quoted"))) for part in slots]
    wanted = source.split(".")
    if len(parts) == 1 and len(wanted) > 1:
        wanted = wanted[-1:]  # a bare name claims the source's own table
    if len(parts) != len(wanted):
        return False
    return all(text == want if quoted else text.lower() == want.lower()
               for (text, quoted), want in zip(parts, wanted))


def _source_aliases(node: exp.Expression, source: str) -> set[str]:
    """The aliases the expression's own subqueries give `source` (`FROM orders r`): a column `r`
    qualifies is a column of the source."""
    return {table.alias.lower() for table in node.find_all(exp.Table)
            if table.alias and _is_source(table, source)}


# A relation `columns_read` does not follow: a CTE, a LATERAL, a VALUES list, an UNNEST. A
# subquery over the source itself it does follow, so only these make an expression foreign.
_FOREIGN_RELATIONS = (exp.With, exp.Lateral, exp.Values, exp.Unnest)


# Constants an engine reads as a column where one is so named; `columns_read` reports their
# spellings. A Boolean is `true` or `false`; the niladic functions their own name.
_KEYWORD_COLUMNS = (exp.Boolean, exp.CurrentDate, exp.CurrentTimestamp, exp.CurrentTime,
                    exp.CurrentUser)


# Any digit-led token of the expression's own text, which MySQL reads as a column so named
# (`1e`, `1E`, `123e`, measured in the dimension tests; Codex's fourth review of #172):
# sqlglot reads `1e` as a number, `1ea` as `1e AS a` and `123abc` as `123 AS abc`, a number
# and an implicit alias, so the parse holds no node that names the column (the review gate on
# the seventh commit). Read off the text, as the spelling fallback reads a bare expression,
# qualified or not (`r.1ea` is `r.1e AS a` to the parser: the gate on the eighth commit), inside
# quoted text and comments too, which withholds more and never less; a token naming no column
# decides nothing. The characters are MySQL's own rule for an unquoted identifier, ASCII
# letters, digits, `_` and `$` and anything in U+0080 to U+FFFF (combining marks and currency
# symbols among them), not Python's word class: `1é` and `1你好` were read by nothing (Codex's
# fifth review of #172), and `1$`, `1€` and the decomposed `1e\u0301x` cut short or missed
# under `\w` (the review gate on the tenth commit). Digits in the lead are ASCII, as MySQL's.
_IDENT_CHAR = r"[A-Za-z0-9_$\u0080-\uffff]"
_DIGIT_LED = re.compile(r"(?<!" + _IDENT_CHAR + r")[0-9]+[A-Za-z_$\u0080-\uffff]" + _IDENT_CHAR + r"*")


def _keyword_spelling(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Boolean):
        return "true" if node.this else "false"
    if node.expressions or node.args.get("this"):
        return None  # `current_date(...)` with arguments is a call, never a column
    return node.sql().split("(")[0].strip().lower() or None


def keyword_columns(expr: str) -> set[str]:
    """The spellings of the keyword constants `expr` holds, anywhere in it, each a column the
    engine reads instead where one is so named (`true` on sqlite, `current_date` on DuckDB,
    measured in the dimension tests). They parse as constants, not columns, so inside a
    subquery `(SELECT max(true) FROM orders)` over a phi column named `true` was read by
    nothing and reached the uncleared (Codex's third review of #172). Kept apart from
    `columns_read`, whose other caller marks a dimension unresolved when a ref names no
    column (the review gate on the first draft): the raw-read check asks these only where
    the table has the column, and a spelling naming none decides nothing."""
    out: set[str] = set()
    for dialect in _EXPR_DIALECTS:
        try:
            node = sqlglot.parse_one(expr, read=dialect)
        except sqlglot.errors.SqlglotError:
            continue
        if node is None:
            continue
        for keyword in node.find_all(*_KEYWORD_COLUMNS):
            spelling = _keyword_spelling(keyword)
            if spelling:
                out.add(spelling)
        break
    out.update(token.lower() for token in _DIGIT_LED.findall(expr))
    return out


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
