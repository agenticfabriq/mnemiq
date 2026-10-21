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
    follows (`r.secret_code.value`, a nested field path), and any dotted path's first part is the
    column it may be as a struct, qualified by the parser or not (`secret_code.value`, and
    `orders.value` over a struct column named like the source: Codex's seventh and eighth
    reviews of #172). Every reading is kept, the whole name a column may truly be named among
    them, and a reference is one set of the names it could be; a name among them that names no
    column decides nothing, so the extra readings withhold more, never less.

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
            # `\`r.secret_code\`` the same on its second). A dotted path's first part is the
            # column it may be as a struct whose field follows, qualified by the parser or not
            # (`\`secret_code.value\``: the seventh review; `orders.value` over `orders r` with a
            # struct column named `orders`, which the parser reads as a qualifier: the eighth).
            # Every prefix is tried, the whole reading kept beside them.
            parts = [p for p in (qualifier.split(".") if qualifier else []) + column.name.split(".") if p]
            if len(parts) > 1:
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


# A relation `columns_read` does not follow: a CTE, a LATERAL, a VALUES list, an UNNEST, a
# PIVOT or UNPIVOT. A subquery over the source itself it does follow, so only these make an
# expression foreign. A PIVOT keeps every column it neither aggregates nor pivots on as a
# grouping column no node names, and its alias is no alias of the source, so `(SELECT
# max(to_json(p)) FROM orders PIVOT (sum(id) FOR status IN ('x')) p)` passed the whole-row
# guard and a phi column's value reached the uncleared inside the JSON (Codex's tenth review
# of #172); an UNPIVOT carries the rest of the row the same way, and DuckDB's own `PIVOT
# orders ON status USING sum(id)` parses to the same node.
_FOREIGN_RELATIONS = (exp.With, exp.Lateral, exp.Values, exp.Unnest, exp.Pivot)


# Any token of the expression's own text, which an engine reads as a column so named whatever
# the parser made of it. A keyword constant parses as a constant (`true` on sqlite,
# `current_date` on DuckDB, measured in the dimension tests; Codex's third review of #172); a
# digit-led name as a number and an implicit alias (`1e` a number, `1ea` is `1e AS a`, `123abc`
# is `123 AS abc`: the fourth review, the gate on the seventh and eighth commits); and an
# argument the parser reads as a unit or a type as a string or a keyword: `date_trunc(
# secret_code, id)` parses as `DATE_TRUNC('SECRET_CODE', id)`, the quoted name the same, as do
# `extract(secret_code FROM id)`, `INTERVAL 1 secret_code`, `last_day(id, secret_code)`,
# `timestamp_trunc(id, secret_code)` and `CAST(id AS secret_code)`, while DuckDB reads the unit
# column, a withheld `year` and `day` giving different dates (Codex's eleventh review). The
# parse holds no node that names the column in any of these, and the parser's next rewrite is
# not known in advance, so every token is read off the text, qualified or not (`r.1ea` is
# `r.1e AS a` to the parser), inside comments too, which withholds more and never less: a
# token naming no column decides nothing. A token is a maximal run of MySQL's
# own characters for an unquoted identifier, ASCII letters, digits, `_` and `$` and anything
# in U+0080 to U+FFFF (combining marks and currency symbols among them: `1é`, `1你好`, `1$`,
# `1€` and the decomposed `1e\u0301x` were missed or cut short under Python's word class; the
# fifth review, the gate on the tenth commit) and beyond it, since sqlglot reads an unquoted
# astral character in a unit position as a string too (`date_trunc(💩, id)`: Codex's twelfth
# review), that is not digits alone, a number to every engine. A QUOTED name is read whole
# besides, with its doubled quotes undoubled: `date_trunc("secret code", id)`, `"secret.code"`,
# `"123"`, `"null"` and `"se""cret"` each parse as the unit string, and the token rule split or
# dropped the name (the same review), so each quoted span of the text, double-quoted,
# backticked or single-quoted (sqlite reads a single-quoted name as the column where no
# string fits), is a column it may be as written. The text is scanned left to right, a
# comment, a quoted span or a token at a time, so that a quote inside a comment pairs within
# the comment alone: scanned for quoted spans alone, `/* " */ "secret code"` paired the
# comment's quote with the name's opening one and the name fell to the token rule, split
# (Codex's thirteenth review of #172). A comment's own text is scanned the same way, since an
# engine may run it as code: MySQL reads `--secret_code` as two minus signs and the column
# (no space after the dashes) and runs `/*! ... */` as SQL, while sqlglot drops both and the
# expression parses as its own (the review gate on the nineteenth commit, where a comment
# contributed nothing). A block comment nests to DuckDB and Postgres and not to MySQL, which
# ends it at the first `*/`: ended there, `/* /* */ " */ "secret code"` left a quote outside
# the comment to pair with the name's opening one (Codex's fourteenth review), while nested, a
# name that opens inside the region and closes after it (`` `secret */ code` ``, read whole by
# MySQL) is split. The text is scanned under both readings and the spellings joined, which
# withholds more and never less; a comment's own text is scanned as code under each, so what
# one reading hides the other reads. An open comment runs to the end of the text.
_IDENT_CHAR = r"[A-Za-z0-9_$\u0080-\uffff\U00010000-\U0010ffff]"
_SCAN = re.compile(r"(?P<line>--[^\n]*)|(?P<block>/\*)"
                   r'|"(?P<double>(?:[^"]|"")+)"|`(?P<back>(?:[^`]|``)+)`|\'(?P<single>(?:[^\']|\'\')+)\''
                   r"|(?P<token>(?![0-9]+(?!" + _IDENT_CHAR + r"))" + _IDENT_CHAR + r"+)", re.S)
_QUOTE = {"double": '"', "back": "`", "single": "'"}
# Whitespace or a comment may stand between the closing quote and UESCAPE, and between it and
# its character: Postgres reads the clause in the grammar, past comments (the review gate on
# the twenty-second commit). An escape is exactly four or six hex digits.
_GAP = r"(?:\s|/\*.*?\*/|--[^\n]*(?:\n|$))*"
_UESCAPE = re.compile(_GAP + r"UESCAPE" + _GAP + r"'(?P<esc>[^'])'", re.I | re.S)
_HEX4 = re.compile(r"[0-9A-Fa-f]{4}")
_HEX6 = re.compile(r"[0-9A-Fa-f]{6}")
# The one spelling every engine reads as the constant unquoted, never as a column so named: a
# dimension `null` over a table with a phi column `null` is shown to the uncleared (the
# dimension tests), where `true` on sqlite and `current_date` on DuckDB read the column.
# Quoted, `"null"` is the column (the review gate on the seventeenth commit): the scan keeps
# a quoted span apart from a token, and the constant is dropped from the tokens alone.
_NEVER_A_COLUMN = frozenset({"null"})


def keyword_columns(expr: str) -> set[str]:
    """The spellings of every token of `expr`'s own text, each a column the engine reads where
    one is so named whatever the parser made of it, a keyword constant, a number, a unit or a
    type among them (`_IDENT_TOKEN`). Kept apart from `columns_read`, whose other caller marks
    a dimension unresolved when a ref names no column (the review gate on the first draft):
    the raw-read check asks these only where the table has the column, and a spelling naming
    none decides nothing. An unquoted `null` is left out, the constant to every engine; a
    quoted name is read whole; a comment's text is scanned as code in its turn (`_SCAN`)."""
    return _scan(expr, nested=True) | _scan(expr, nested=False)


def _scan(expr: str, nested: bool) -> set[str]:
    """`keyword_columns` under one reading of a block comment: nested (DuckDB, Postgres) or
    ended at the first `*/` (MySQL)."""
    out: set[str] = set()
    pos = 0
    while (match := _SCAN.search(expr, pos)) is not None:
        kind, pos = match.lastgroup, match.end()
        if kind == "token":
            if match.group().lower() not in _NEVER_A_COLUMN:
                out.add(match.group().lower())
        elif kind in _QUOTE:
            span = match.group(kind).replace(_QUOTE[kind] * 2, _QUOTE[kind])
            out.add(span.lower())
            decoded = _unicode_escaped(expr, match, span)
            if decoded is not None:
                out.add(decoded.lower())
        elif kind == "line":
            out |= _scan(match.group()[2:], nested)
        else:
            pos, closed = _block_end(expr, match.start(), nested)
            out |= _scan(expr[match.start() + 2:pos - 2 if closed else pos], nested)
    return out


def _unicode_escaped(expr: str, match: re.Match, span: str) -> str | None:
    """The name a Postgres Unicode-escaped span decodes to, or None where the span is not one
    or the escapes are not Postgres's: `U&` touching the opening quote marks it (`U& "x"` is
    the `&` operator over a column `U`, which DuckDB binds), the escape character `\\` unless
    a `UESCAPE 'c'` follows the closing quote, `\\XXXX` and `\\+XXXXXX` a code point each,
    the escape doubled the character itself, a surrogate pair one character. `U&"secret\\005fcode"`
    is `secret_code` to Postgres (Codex's fifteenth review of #172), where the spellings
    read were `u` and the span as written and the phi column so named was offered through
    the metric and the dimension selectors alike; DuckDB refuses the spelling, MySQL and
    SQLite read the operator. The span as written stays a spelling too."""
    if expr[max(match.start() - 2, 0):match.start()].lower() != "u&":
        return None
    tail = _UESCAPE.match(expr, match.end())
    esc = tail.group("esc") if tail else "\\"
    out, i = [], 0
    try:
        while i < len(span):
            if span[i] != esc:
                out.append(span[i])
                i += 1
            elif span.startswith(esc, i + 1):
                out.append(esc)
                i += 2
            elif span[i + 1:i + 2] == "+" and _HEX6.fullmatch(span[i + 2:i + 8]):
                out.append(chr(int(span[i + 2:i + 8], 16)))
                i += 8
            elif _HEX4.fullmatch(span[i + 1:i + 5]):
                out.append(chr(int(span[i + 1:i + 5], 16)))
                i += 5
            else:
                return None  # an escape Postgres refuses: no name is read
        return "".join(out).encode("utf-16", "surrogatepass").decode("utf-16")
    except (ValueError, UnicodeError):
        return None  # a code point or a lone surrogate no name holds


def _block_end(text: str, start: int, nested: bool) -> tuple[int, bool]:
    """Where the block comment opening at `start` ends, and whether a `*/` closed it: after
    the `*/` that brings its depth back to zero where comments nest, after the first `*/`
    where they do not, or at the end of the text where none does (the review gate on the
    twenty-first commit: cut as if closed, an open comment lost its last two characters,
    `/*! + secret_code */` inside one reading `secret_cod`)."""
    depth, i = 0, start
    while i < len(text):
        if text.startswith("/*", i) and (nested or i == start):
            depth, i = depth + 1, i + 2
        elif text.startswith("*/", i):
            depth, i = depth - 1, i + 2
            if depth == 0:
                return i, True
        else:
            i += 1
    return len(text), False


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
                not (isinstance(star.parent, exp.Count) and not any(star.args.values()))
                for star in node.find_all(exp.Star)):
            # A star reads columns no node names, so `columns_read` cannot report them: `(SELECT
            # max(q.secret_code) FROM (SELECT * FROM orders) q)` reported nothing and passed the
            # uncleared (the review gate on the first draft). A bare COUNT(*) alone reads none;
            # a star with a modifier (`count(* EXCLUDE (id))`) reads the rest of the row, and
            # rode the exemption to the uncleared (Codex's ninth review of #172).
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
