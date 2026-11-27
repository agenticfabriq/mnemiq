"""The rows of a dashboard plan whose columns are named with their tables (`order_items.revenue`, `customers.region`):
each measure's base table, the chain of LEFT JOINs from that base to every other table a part needs, and the name the
joined dataset gives each column. Nothing here reads a source or Superset; the planner reads a draft through it.

A measure's base is the table whose rows it adds up: the table of the columns its aggregates add up, and for one that
adds up no column (a count or a share of 1s), the table it gives, else the one table its conditions name (`base_of`). The
other tables a part needs are joined to the base along links, each an equality of one column of each table: the links
the model states, and for a pair of tables it left unlinked, the snapshot's relationships (`edges`). A table is joined
by a link only where the link's columns on that table, its far side, hold each value once, so each of the base's rows
meets at most one of its rows, and LEFT JOIN keeps every base row: the measure's value is the same whichever tables a
chart joins. That holds only where the equality compares the two columns exactly: one that converts a side may read
two of the far side's values as one (a bigint read as a double precision rounds two ids past 2^53 to one), and a key
that holds each value once then meets a row twice. So a link joins two columns of one kind (`comparable`), whatever is
known of its far side, and one of two kinds, or with a column of no recorded type, is never joined by (`mistyped`). A
far side known to repeat is never joined; one not known either way may be (`far_side`), and the plan check measures it
before the plan is accepted. Each needed table is reached by its shortest chain of such links
(`chain`); two chains of one length, or two links between the same two tables, are refused as ambiguous, since each
gives the measure different rows to read, unless the links the model stated tell the two chains apart. A part that adds
nothing up takes as its rows the one of its tables that reaches the others (`base_among`).

The repair shows the model its plan in the form it wrote it: `read_join` reads back a join `join_text` wrote, `qualify`
names each of a dataset's columns with its table again, as `rewrite` named it the other way, and `column_text` writes
one column as `column_of` reads it.

Every identifier is read as Superset mode's dialect (`superset_dialect`) reads it (`resolve_identifier`): an unquoted
name folds as the dialect folds it, and a quoted one is the name as written. Every identifier built here is quoted, so
it reads back as the stored name it was built from; a column whose name the joined dataset keeps is left as the model
wrote it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.contract.semantic import Relationship
from mnemiq.sql.fanout_check import value_terms
from mnemiq.sql.identifiers import resolve_identifier
from mnemiq.superset.measures import Aggregation, aggregates
from mnemiq.superset.settings import superset_dialect

# Whether a table's column holds each value once, by (table, column): enrichment's key facts. A column absent is
# unknown.
Unique = Mapping[tuple[str, str], bool]
# Each column's declared type, by (table, column), as the snapshot records it (`column_types`). A column absent has no
# recorded type.
Types = Mapping[tuple[str, str], str | None]

_T = exp.DataType.Type
# The types an equality compares exactly with another type, by the kind they share: Postgres compares an integer with an
# exact numeric as numeric, which rounds neither, and text with varchar as text; `bpchar` is its own name for char. Any
# other type compares exactly only with itself.
_KINDS = {_T.SMALLINT: "exact number", _T.INT: "exact number", _T.BIGINT: "exact number", _T.DECIMAL: "exact number",
          _T.TEXT: "text", _T.VARCHAR: "text", _T.CHAR: "blank-padded text", _T.BPCHAR: "blank-padded text"}
# Oracle's built-in types, read from the text its data dictionary records for a column of one
# (`ALL_TAB_COLUMNS.DATA_TYPE`, with a length or a precision where one is given), and the spellings a column is
# declared with that Oracle stores as a NUMBER, each by its leading words in any case, as Oracle reads a word it is not
# given quoted: never by sqlglot's reading, which takes a name Oracle has no type of for one of its own built-ins
# (GEOMETRY, TEXT, UUID, DATETIME), CLOB for its TEXT and LONG for its BIGINT, and BINARY_FLOAT and Oracle's FLOAT, a
# NUMBER of binary precision, for one FLOAT. Any other text is no type Oracle records as a built-in: an object type,
# Oracle's own (SDO_GEOMETRY, XMLTYPE) or the schema's, of whatever name, which compares by rules of its own (a MAP
# method's value can read two objects as one), or UNDEFINED, Oracle's word for a type it cannot describe.
_ORACLE_LENGTH = r"(?:\s*\([\w\s,*]*\))?"
_ORACLE_PRECISION = r"(?:\s*\(\s*\d+\s*\))?"
_ORACLE_BUILT_IN = re.compile(
    rf"\s*(?:(?P<sized>NUMBER|FLOAT|DECIMAL|NUMERIC|VARCHAR2|VARCHAR|NVARCHAR2|CHAR|NCHAR|RAW|UROWID|VECTOR)"
    rf"{_ORACLE_LENGTH}"
    r"|(?P<long>LONG(?:\s+RAW)?)"
    r"|(?P<plain>INTEGER|INT|SMALLINT|BINARY_FLOAT|BINARY_DOUBLE|DATE|ROWID|CLOB|NCLOB|BLOB|BFILE|JSON|BOOLEAN)"
    rf"|(?P<timestamp>TIMESTAMP){_ORACLE_PRECISION}(?P<zone>\s+WITH(?:\s+LOCAL)?\s+TIME\s+ZONE)?"
    rf"|INTERVAL\s+(?:(?P<year>YEAR){_ORACLE_PRECISION}\s+TO\s+MONTH"
    rf"|(?P<day>DAY){_ORACLE_PRECISION}\s+TO\s+SECOND{_ORACLE_PRECISION}))\s*",
    re.IGNORECASE)
# Oracle's built-in types by the kind they share: NUMBER with INTEGER, INT, SMALLINT, DECIMAL and NUMERIC, which Oracle
# stores as NUMBERs and compares as NUMBER, which rounds neither; VARCHAR2 and VARCHAR, compared as text without
# padding; and CHAR, compared padded with blanks. NVARCHAR2 and NCHAR, in the national character set, are each a kind
# of their own: Oracle compares a VARCHAR2 or a CHAR with one by converting it into that set, which reads every byte
# the database character set cannot decode as one replacement character, and in some legacy multibyte sets several
# characters as one, so two keys held once each can equal one value. Any other built-in type compares exactly only
# with itself, its length aside: FLOAT, BINARY_FLOAT and BINARY_DOUBLE each its own, DATE, a TIMESTAMP with a
# TIMESTAMP of the same time zone kind, each kind of INTERVAL, RAW, compared byte for byte, and the row ids.
_ORACLE_KINDS = {"NUMBER": "exact number", "INTEGER": "exact number", "INT": "exact number",
                 "SMALLINT": "exact number", "DECIMAL": "exact number", "NUMERIC": "exact number",
                 "VARCHAR2": "text", "VARCHAR": "text", "NVARCHAR2": "national text", "CHAR": "blank-padded text",
                 "NCHAR": "national blank-padded text"}
# Oracle's built-in types it refuses in a comparison: its LOBs and LONGs (LONG RAW among them), JSON, VECTOR and
# BOOLEAN.
_ORACLE_UNCOMPARED = frozenset({"CLOB", "NCLOB", "BLOB", "BFILE", "LONG", "LONG RAW", "JSON", "VECTOR", "BOOLEAN"})
# The kind of a type no equality compares, not even with itself.
_UNCOMPARED = "compared by no equality"


@dataclass(frozen=True)
class Link:
    """One column of one table equal to one column of another, each as (table, column). A link of several columns is
    several Links between the same two tables with one `via`: the relationship's id for a snapshot's link, and for a
    stated one its tables as `<from table>-><to table>`. Two `via`s between one pair of tables are two links."""

    left: tuple[str, str]
    right: tuple[str, str]
    via: str = ""


@dataclass(frozen=True)
class Step:
    """One LEFT JOIN of a chain: `table`, joined on `on`, each link's left a table already joined and its right
    `table`."""

    table: str
    on: tuple[Link, ...]


@dataclass(frozen=True)
class Refusal:
    """Why a part's rows cannot be derived, in words the model can repair the part from."""

    reason: str


def links_of(relationships: Sequence[Relationship], names: Mapping[str, str]) -> tuple[Link, ...]:
    """The links the snapshot's relationships give, one for each key column, between tables named as `names` names
    each snapshot object id, each with its relationship's id as its `via`. A relationship touching an object `names`
    does not hold, a table the builder cannot see, gives none."""
    return tuple(Link((names[relationship.from_], key.left), (names[relationship.to], key.right), relationship.id)
                 for relationship in relationships if relationship.from_ in names and relationship.to in names
                 for key in relationship.join_keys)


def edges(stated: Sequence[Link], known: Sequence[Link]) -> dict[frozenset[str], tuple[Link, ...]]:
    """The links joining each pair of tables: those the model stated for the pair, where it stated any, else those the
    snapshot knows. A column pair given twice in one link, or once from each end, is one; two links of the same
    columns are one link; a link from a table to itself joins nothing and is left out."""
    return {**_by_pair(known), **_by_pair(stated)}


def _by_pair(links: Sequence[Link]) -> dict[frozenset[str], tuple[Link, ...]]:
    grouped: dict[frozenset[str], dict[str, dict[frozenset[tuple[str, str]], Link]]] = {}
    for link in links:
        pair = frozenset((link.left[0], link.right[0]))
        if len(pair) == 2:
            grouped.setdefault(pair, {}).setdefault(link.via, {}).setdefault(frozenset((link.left, link.right)), link)
    paired: dict[frozenset[str], tuple[Link, ...]] = {}
    for pair, by_via in grouped.items():
        kept: list[Link] = []
        seen: set[frozenset[frozenset[tuple[str, str]]]] = set()
        for columns in by_via.values():
            if frozenset(columns) not in seen:
                seen.add(frozenset(columns))
                kept.extend(columns.values())
        paired[pair] = tuple(kept)
    return paired


def _one_expression(expression: str) -> exp.Select | None:
    """`expression` parsed as the one column of a SELECT, or None when it is not one SQL expression: one the parser
    refuses, or one nested deeper than it reads (a RecursionError), which a model's text can be."""
    try:
        tree = sqlglot.parse_one(f"SELECT {expression}", read=superset_dialect())
    except (SqlglotError, RecursionError):
        return None
    if not isinstance(tree, exp.Select) or len(tree.expressions) != 1 \
            or {key for key, value in tree.args.items() if value} - {"expressions"}:
        return None
    return tree


def qualified(expression: str) -> list[tuple[str, str]] | str:
    """Each column `expression` names, as (table, column), once each in the order written; or why it cannot be read: it
    is not one SQL expression, it holds a subquery, or it names a column without its table (or with a schema, or
    `table.*`)."""
    tree = _one_expression(expression)
    if tree is None:
        return f"{expression!r} is not one SQL expression"
    if any(node is not tree for node in tree.find_all(exp.Select)) or tree.find(exp.Table):
        return f"{expression!r} holds a subquery: name each column with its table, as <table>.<column>"
    named: dict[tuple[str, str], None] = {}
    for column in tree.find_all(exp.Column, bfs=False):
        written = column.sql(dialect=superset_dialect())
        if isinstance(column.this, exp.Star):
            return f"{written} is not one column: name each column with its table, as <table>.<column>"
        if column.args.get("db") or column.args.get("catalog"):
            return f"{written} names a schema: name each column with its table alone, as <table>.<column>"
        if not column.table:
            return f"{column.name!r} is named without its table: write it as <table>.{written}"
        named[(resolve_identifier(column.args["table"], superset_dialect()),
               resolve_identifier(column.this, superset_dialect()))] = None
    return list(named)


def column_of(text: str) -> tuple[str, str] | str:
    """The (table, column) `text` names when it is one column named with its table, `<table>.<column>`, as a chart's
    dimension, time column, filter or a link's side is; or why it is not."""
    named = qualified(text)
    if isinstance(named, str):
        return named
    tree = _one_expression(text)
    if tree is None or not isinstance(tree.expressions[0], exp.Column):
        return f"{text!r} is not one column: name one column with its table, as <table>.<column>"
    return named[0]


def _written(name: str) -> exp.Identifier:
    """`name` as an identifier Superset mode's dialect reads back as `name`: unquoted where it reads so, else quoted."""
    plain = exp.to_identifier(name, quoted=False)
    tree = _one_expression(plain.sql(dialect=superset_dialect()))
    read = tree.expressions[0] if tree is not None else None
    if isinstance(read, exp.Column) and not read.table and resolve_identifier(read.this, superset_dialect()) == name:
        return plain
    return exp.to_identifier(name, quoted=True)


def column_text(table: str, column: str) -> str:
    """`<table>.<column>` as the model writes a column, each name unquoted where the dialect reads it back so: the text
    `column_of` reads as (table, column)."""
    return exp.Column(this=_written(column), table=_written(table)).sql(dialect=superset_dialect())


def aggregated(expression: str) -> set[str]:
    """The tables of the columns `expression`'s aggregates add up: each aggregate's value columns (`value_terms`), so a
    CASE's condition, which only chooses rows, is not among them, and an ordered-set aggregate's are those it orders
    by. Empty when `expression` is not one SQL expression (`qualified` says why)."""
    tree = _one_expression(expression)
    if tree is None:
        return set()
    tables: set[str] = set()
    for aggregate in tree.find_all(exp.AggFunc):
        # `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` holds its value, x, beside the function, not in it.
        within = aggregate.parent if isinstance(aggregate.parent, exp.WithinGroup) else aggregate
        for term in value_terms(within):
            tables.update(resolve_identifier(column.args["table"], superset_dialect()) for column in term
                          if column.table)
    return tables


def _named_in_aggregates(expression: str) -> set[str]:
    """The tables of every column `expression`'s aggregates name: in what they add up and in what chooses their rows, a
    CASE's condition, a FILTER's WHERE and the like; never a window's PARTITION BY or ORDER BY, which chooses no rows of
    the aggregate it is over. Empty when `expression` is not one SQL expression (`qualified` says why)."""
    tree = _one_expression(expression)
    if tree is None:
        return set()
    tables: set[str] = set()
    for aggregate in tree.find_all(exp.AggFunc):
        # `COUNT(*) FILTER (WHERE x)` holds its condition, and `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` its
        # value, beside the function, not in it.
        scope: exp.Expression = aggregate
        while isinstance(scope.parent, (exp.Filter, exp.WithinGroup)) and scope.parent.this is scope:
            scope = scope.parent
        tables.update(resolve_identifier(column.args["table"], superset_dialect())
                      for column in scope.find_all(exp.Column) if column.table)
    return tables


# Why an expression that gives one value per row is no measure, whatever its `table`: read as the plan check reads it
# (`measures.aggregates`), so the draft goes back with the check's reason, never told to give a table.
_NO_AGGREGATE = ("it is no aggregate: it gives one value per row, not one per group; write the measure with an "
                 "aggregate, as SUM(<table>.<column>) or COUNT(*)")
_WINDOWED = ("its aggregate is computed over a window (OVER), which gives one value per row, not one per group, so it "
             "is no measure either: write the aggregate without OVER")
_UNMODELLED = ("mnemiq cannot tell whether it aggregates: it holds a function mnemiq does not model and no aggregate "
               "mnemiq does; write the measure with SQL's own aggregates, as SUM(<table>.<column>) or COUNT(*)")
# Why a measure whose aggregates name no column, `COUNT(*)` or `SUM(1)`, needs its `table`.
_NAMES_NO_COLUMN = ('it counts rows without saying whose: its aggregate names no column, so give the measure "table", '
                    "the table whose rows it counts, or count one of that table's columns, as COUNT(<table>.<key>)")


def base_of(expression: str, table: str | None) -> str | Refusal:
    """The table whose rows a written measure adds up: the one table of the columns its aggregates add up. Where they
    add up no column, `table`, the table the measure names; without one, the one table of every column its aggregates
    name (`_named_in_aggregates`), so a count, sum or share of 1s counts the rows of the table its condition names.
    `table` decides nothing for an expression whose aggregates add up a column. Refused when the expression cannot be
    read (`qualified`); when it holds no aggregate, or only a window's, so it gives one value per row
    (`measures.aggregates`, as the plan check reads it), or holds no aggregate but a function mnemiq does not model,
    whatever its `table`; when its aggregates add up columns of two tables; and, without `table`, when they add up
    none and name columns of no table (`COUNT(*)`) or of two."""
    named = qualified(expression)
    if isinstance(named, str):
        return Refusal(named)
    shape = aggregates(expression)
    if shape is not Aggregation.AGGREGATES:
        tree = _one_expression(expression)
        held = tree is not None and tree.find(exp.AggFunc) is not None
        if shape is Aggregation.ROW_LEVEL:
            return Refusal(_WINDOWED if held else _NO_AGGREGATE)
        if not held:
            return Refusal(_UNMODELLED)
    tables = sorted(aggregated(expression))
    if len(tables) > 1:
        return Refusal(f"it adds up columns of {_listed(tables)}, and a measure adds up the rows of one table: plan "
                       "one measure for each table")
    if tables:
        return tables[0]
    if table is not None:
        return table
    chosen = sorted(_named_in_aggregates(expression))
    if len(chosen) == 1:
        return chosen[0]
    if chosen:
        return Refusal(f"it adds up no column, and its conditions name columns of {_listed(chosen)}, so it does not say "
                       'whose rows it counts: give the measure "table", the table whose rows it counts')
    return Refusal(_NAMES_NO_COLUMN)


def far_side(table: str, columns: Sequence[str], unique: Unique) -> bool | None:
    """Whether `columns` of `table`, a link's far side, hold each value once together: True when any of them is known
    to, False only when they are one column known to repeat, else None, unknown. Several columns each known to repeat
    are unknown, since together they may still hold each value once."""
    distinct = list(dict.fromkeys(columns))
    known = [unique.get((table, column)) for column in distinct]
    if any(fact is True for fact in known):
        return True
    if len(distinct) == 1 and known[0] is False:
        return False
    return None


def _oracle_kind(data_type: str) -> tuple[str, str]:
    """The kind of a column of `data_type` on Oracle, read from the type text's leading words (`_ORACLE_BUILT_IN`),
    every length and precision aside: its kind in `_ORACLE_KINDS`, else its own type; and the kind no equality compares
    for one of `_ORACLE_UNCOMPARED`, and for any text Oracle records as no built-in, which is said as written."""
    match = _ORACLE_BUILT_IN.fullmatch(data_type)
    if match is None:
        return _UNCOMPARED, data_type.strip()
    if match["timestamp"]:
        return "type", " ".join(["TIMESTAMP", *(match["zone"] or "").upper().split()])
    if match["year"] or match["day"]:
        return "type", "INTERVAL YEAR TO MONTH" if match["year"] else "INTERVAL DAY TO SECOND"
    name = " ".join((match["sized"] or match["long"] or match["plain"]).upper().split())
    if name in _ORACLE_UNCOMPARED:
        return _UNCOMPARED, name
    if name in _ORACLE_KINDS:
        return "kind", _ORACLE_KINDS[name]
    return "type", name


def _kind(data_type: str | None) -> tuple[str, str] | None:
    """The kind of a column of `data_type`, as the snapshot records it, read as Superset mode's dialect reads a type,
    every length and precision aside (`varchar(10)` is a varchar): its kind in the dialect's kinds (`_KINDS`), else the
    type itself; on Oracle, as Oracle's data dictionary records a built-in type (`_oracle_kind`), the kind no equality
    compares for any other text. None for no type, or one that names none: unknown, a user-defined type of no name, or
    an array of elements of no type."""
    if data_type is None or not data_type.strip():
        return None
    dialect = superset_dialect()
    if dialect == "oracle":
        return _oracle_kind(data_type)
    try:
        built = exp.DataType.build(data_type, dialect=dialect, udt=True)
    except (SqlglotError, ValueError):
        return None
    for argument in list(built.find_all(exp.DataTypeParam)):
        argument.pop()
    if built.this == _T.UNKNOWN or (built.this == _T.USERDEFINED and not built.args.get("kind")) \
            or (built.this in exp.DataType.NESTED_TYPES and not built.expressions):
        return None
    if built.this in _KINDS and not built.expressions:
        return "kind", _KINDS[built.this]
    return "type", built.sql(dialect=dialect)


def comparable(left_type: str | None, right_type: str | None) -> bool:
    """Whether an equality of a column of `left_type` and one of `right_type` compares their values exactly, so it reads
    no two values of either as one: both an integer or an exact numeric (smallint, integer, bigint, numeric), both text
    or varchar, or both of one other type (char with char, a double precision with a double precision, a timestamp
    with a timestamp of the same time zone kind), each read as Superset mode's dialect reads a type, its length and
    precision aside. On Oracle: both a NUMBER (INTEGER, INT, SMALLINT, DECIMAL and NUMERIC among them), both a VARCHAR2
    or VARCHAR, both an NVARCHAR2, both a CHAR, both an NCHAR, or both of one other built-in type (BINARY_FLOAT,
    BINARY_DOUBLE and FLOAT each its own; DATE; a TIMESTAMP with a TIMESTAMP of the same time zone kind; each kind of
    INTERVAL; RAW, ROWID and UROWID each its own). False where either has no type, or one that names none, and for a
    type no equality compares (Oracle's LOBs, LONGs, JSON, VECTOR and BOOLEAN, and any type its data dictionary records
    as no built-in: its object types and UNDEFINED), whatever the other."""
    kind = _kind(left_type)
    return kind is not None and kind[0] != _UNCOMPARED and kind == _kind(right_type)


def _typed(column: tuple[str, str], data_type: str | None) -> str:
    named = f"{column[0]}.{column[1]}"
    return f"{named} is {data_type.strip()}" if data_type and data_type.strip() else f"{named} has no recorded type"


def mistyped(link: Link, types: Types) -> str | None:
    """Why `link` is never joined by, its columns' types (`types`) not comparing exactly (`comparable`): said with both
    columns and both types, and which of the types no equality compares, for the repair. None where they compare
    exactly."""
    left, right = types.get(link.left), types.get(link.right)
    if comparable(left, right):
        return None
    said = (f"{_typed(link.left, left)} and {_typed(link.right, right)}: a link joins columns whose values compare "
            "exactly")
    kinds = (_kind(left), _kind(right))
    if None in kinds:
        said += ", which mnemiq cannot tell without both columns' types"
    elif never := list(dict.fromkeys(text.strip() for text, kind in zip((left, right), kinds)
                                     if kind[0] == _UNCOMPARED)):
        said += (f", and a column of type {' or '.join(never)} cannot be compared exactly, not even with one of its own "
                 "type")
    return said


def _oriented(links: Sequence[Link], near: str) -> tuple[Link, ...]:
    """`links` each written from `near`, the table already joined, to the table joined by them."""
    return tuple(link if link.left[0] == near else Link(link.right, link.left, link.via) for link in links)


def _ways(links: Sequence[Link], near: str) -> list[tuple[Link, ...]]:
    """The links of one pair of tables written from `near`, each link whole: its columns, the Links of one `via`."""
    by_via: dict[str, list[Link]] = {}
    for link in _oriented(links, near):
        by_via.setdefault(link.via, []).append(link)
    return [tuple(columns) for columns in by_via.values()]


def _link_text(link: Link) -> str:
    return f"{link.left[0]}.{link.left[1]} = {link.right[0]}.{link.right[1]}"


def _listed(names: Sequence[str]) -> str:
    quoted = [repr(name) for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def chain(base: str, needed: Collection[str], edges: Mapping[frozenset[str], Sequence[Link]],
          unique: Unique, stated: Collection[Link] = (), *, types: Types) -> tuple[Step, ...] | Refusal:
    """The LEFT JOINs that bring each of `needed` to `base`'s rows, in join order: for each needed table, the shortest
    chain of links from `base` whose columns compare exactly (`mistyped`, read from `types` before anything is read of
    a far side) and over which each joined table's far side is not known to repeat (`far_side`). Where two shortest
    chains reach a table, the one with the most links the model stated (`stated`, as `edges` read them) is joined,
    when it alone has the most. Refused, naming them, when a needed table is reached by no such chain; when one is
    reached by two chains of one length that no stated link tells apart, or by a step whose two tables two such links
    join (an employee's department and the department an employee manages), since either gives the base's rows other
    rows to read."""
    adjacent: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for pair in edges:
        for near, far in (sorted(pair), sorted(pair, reverse=True)):
            adjacent.setdefault(near, []).append((far, pair))
    # A breadth-first walk from the base: each table's depth, every table one shorter that reaches it, and the links
    # each such step may join by; and each link never joined by, with why.
    depth, parents, order = {base: 0}, {base: []}, [base]
    usable: dict[tuple[str, str], list[tuple[Link, ...]]] = {}
    blocked: list[tuple[tuple[Link, ...], str]] = []
    frontier = [base]
    while frontier:
        reached: list[str] = []
        for near in frontier:
            for far, pair in sorted(adjacent.get(near, ())):
                if far in depth and depth[far] <= depth[near]:
                    continue
                joinable: list[tuple[Link, ...]] = []
                for on in _ways(edges[pair], near):
                    why = next((said for link in on if (said := mistyped(link, types)) is not None), None)
                    if why is None and far_side(far, [link.right[1] for link in on], unique) is False:
                        columns = ", ".join(dict.fromkeys(f"{link.right[0]}.{link.right[1]}" for link in on))
                        why = (f"the link {' and '.join(map(_link_text, on))} would repeat rows of {on[0].left[0]!r}, "
                               f"since {columns} holds a value more than once")
                    if why is None:
                        joinable.append(on)
                    else:
                        blocked.append((on, why))
                if not joinable:
                    continue
                if far not in depth:
                    depth[far], parents[far] = depth[near] + 1, []
                    order.append(far)
                    reached.append(far)
                parents[far].append(near)
                usable[(near, far)] = joinable
        frontier = reached

    wanted = sorted(set(needed) - {base})
    missing = [table for table in wanted if table not in depth]
    if missing:
        reason = (f"{_listed(missing)} cannot be joined to the rows of {base!r}: no link reaches "
                  f"{'it' if len(missing) == 1 else 'them'} from {base!r}, directly or through other tables, whose "
                  "columns compare exactly and, on each table it joins, hold each value once")
        for on, why in blocked:
            if on[0].right[0] in missing:
                reason += f"; {why}"
        reason += ('; if a link exists, state it in "links": from the referencing table\'s column to the referenced '
                   'table\'s key, as {"from": "<table>.<column>", "to": "<table>.<column>"}')
        return Refusal(reason)

    # Each table's parents on its shortest chains with the most stated links, walked in breadth-first order so each
    # parent's count is known first: a step is stated when the model stated a link between its two tables.
    pairs = {frozenset((link.left[0], link.right[0])) for link in stated}
    most: dict[str, int] = {base: 0}
    chosen: dict[str, list[str]] = {base: []}
    for table in order[1:]:
        counts = {parent: most[parent] + (frozenset((parent, table)) in pairs) for parent in parents[table]}
        most[table] = max(counts.values())
        chosen[table] = [parent for parent in parents[table] if counts[parent] == most[table]]

    def path_to(table: str) -> list[str]:
        """`table`'s chain from the base, through each table's first chosen parent."""
        path = [table]
        while chosen[path[-1]]:
            path.append(chosen[path[-1]][0])
        return path[::-1]

    def path_text(path: Sequence[str]) -> str:
        return " then ".join(_link_text(link) for near, far in zip(path, path[1:]) for link in usable[(near, far)][0])

    joined: set[str] = set()
    for table in wanted:
        walked = [table]
        while chosen[walked[-1]]:
            node = walked[-1]
            if len(chosen[node]) > 1:
                rest = walked[::-1][1:]
                first, second = (path_to(parent) + [node] + rest for parent in chosen[node][:2])
                return Refusal(f"{table!r} is reached from the rows of {base!r} two ways of one length, by "
                               f"{path_text(first)}, and by {path_text(second)}; mnemiq joins a table one way only: "
                               "state the links of the way you mean")
            near = chosen[node][0]
            ways = usable[(near, node)]
            # One link uses each of its columns once: one that uses a column twice is two links, as a buyer's and a
            # payer's both equal to one customer's key.
            if len(ways) == 1 and any(len({getattr(link, side) for link in ways[0]}) < len(ways[0])
                                      for side in ("left", "right")):
                ways = [(link,) for link in ways[0]]
            if len(ways) > 1:
                named = " or ".join(" and ".join(map(_link_text, on)) for on in ways)
                return Refusal(f"{near!r} reaches {node!r} by more than one link, {named}: state only the one to join "
                               "by as a link")
            walked.append(near)
        joined.update(walked)
    return tuple(Step(table, usable[(chosen[table][0], table)][0])
                 for table in order if table in joined and table != base)


def base_among(tables: Collection[str], edges: Mapping[frozenset[str], Sequence[Link]], unique: Unique,
               stated: Collection[Link] = (), *, types: Types) -> str | Refusal:
    """The rows of a part that adds nothing up (a chart with no measure): the one of `tables`, the tables its columns
    are of, from whose rows `chain` reaches every other. Refused, naming them, when none does, or when two do, since
    the part does not say whose rows it shows."""
    if not tables:
        return Refusal("it names no column, so it says no table for its rows")
    named = sorted(tables)
    bases = [table for table in named
             if not isinstance(chain(table, tables, edges, unique, stated, types=types), Refusal)]
    if len(bases) == 1:
        return bases[0]
    if not bases:
        return Refusal(f"its columns are of {_listed(named)}, and the rows of none of these reach the others by links "
                       "whose columns compare exactly and, on each table they join, hold each value once: plot a "
                       "measure on the table whose rows it shows, or state the link that joins them")
    return Refusal(f"its columns are of {_listed(named)}, and the rows of each of {_listed(bases)} reach the others, so "
                   "which rows it shows is not said: plot a measure on the table whose rows it shows")


def _identifier(name: str) -> exp.Identifier:
    return exp.to_identifier(name, quoted=True)


def _column(table: str, column: str) -> exp.Column:
    return exp.Column(this=_identifier(column), table=_identifier(table))


def join_text(base: str, steps: Sequence[Step]) -> str:
    """The FROM clause `steps` describe: `base` LEFT JOIN each step's table ON its links' columns being equal, every
    name quoted as Superset mode's dialect writes it."""
    text = exp.Table(this=_identifier(base)).sql(dialect=superset_dialect())
    for step in steps:
        on = exp.and_(*(exp.EQ(this=_column(*link.left), expression=_column(*link.right)) for link in step.on))
        text += " " + exp.Join(this=exp.Table(this=_identifier(step.table)), on=on, side="LEFT").sql(
            dialect=superset_dialect())
    return text


def read_join(text: str) -> tuple[str, tuple[Step, ...]] | None:
    """The base and the steps of a FROM clause, as `join_text` writes one: its first table, and each table joined
    after it with the equalities of its ON condition between a column of that table and one of a table before it, each
    link written from the table before. A join written another way is read as far as its tables and such equalities go;
    None when `text` is not tables joined, each named alone or with an alias."""
    try:
        tree = sqlglot.parse_one(f"SELECT 1 FROM {text}", read=superset_dialect())
    except SqlglotError:
        return None
    if not isinstance(tree, exp.Select) or tree.args.get("from_") is None:
        return None
    sources = [tree.args["from_"].this, *(join.this for join in tree.args.get("joins") or [])]
    if not all(isinstance(source, exp.Table) for source in sources):
        return None
    tables = [resolve_identifier(source.this, superset_dialect()) for source in sources]
    by_qualifier = {resolve_identifier(source.args["alias"].this if source.args.get("alias") else source.this,
                                       superset_dialect()): table for source, table in zip(sources, tables)}

    def side(column: exp.Expression) -> tuple[str, str] | None:
        if not isinstance(column, exp.Column) or not column.table:
            return None
        table = by_qualifier.get(resolve_identifier(column.args["table"], superset_dialect()))
        return None if table is None else (table, resolve_identifier(column.this, superset_dialect()))

    steps: list[Step] = []
    for index, join in enumerate(tree.args.get("joins") or [], 1):
        on: list[Link] = []
        condition = join.args.get("on")
        for equal in condition.find_all(exp.EQ) if condition is not None else ():
            left, right = side(equal.this), side(equal.expression)
            if left is None or right is None or tables[index] not in (left[0], right[0]) or left[0] == right[0]:
                continue
            on.append(Link(left, right) if right[0] == tables[index] else Link(right, left))
        steps.append(Step(tables[index], tuple(on)))
    return tables[0], tuple(steps)


def qualify(expression: str, owners: Mapping[str, tuple[str, str]]) -> str:
    """`expression`, written over a dataset's columns, with each column it names written as the model writes it, with
    its table: `owners` gives the (table, column) each of the dataset's column names reads (`rewrite`'s map, turned
    round). A column whose name is its own keeps the identifier written; a column named with a table, or one `owners`
    does not hold, is left as it is, and so is `expression` when it is not one SQL expression."""
    tree = _one_expression(expression)
    if tree is None:
        return expression
    for column in list(tree.find_all(exp.Column)):
        if column.table or not isinstance(column.this, exp.Identifier):
            continue
        owner = owners.get(resolve_identifier(column.this, superset_dialect()))
        if owner is None:
            continue
        table, name = owner
        kept = resolve_identifier(column.this, superset_dialect()) == name
        column.replace(exp.Column(this=column.this.copy() if kept else _written(name), table=_written(table)))
    return tree.expressions[0].sql(dialect=superset_dialect())


def rewrite(expression: str, names: Mapping[tuple[str, str], str]) -> str:
    """`expression` with each (table, column) it names written as the joined dataset's name for it in `names`, without
    a table. A column the dataset names as its table does keeps the identifier written, quoting and all, so the
    measure reads as one written over that table alone (`SUM(order_items.revenue)` is `SUM(revenue)`); a renamed one
    is quoted, so it reads back as the name given. `expression` is one `qualified` reads, and `names` names each of its
    columns; a column it does not name is a ValueError."""
    tree = sqlglot.parse_one(f"SELECT {expression}", read=superset_dialect())
    for column in list(tree.find_all(exp.Column)):
        key = (resolve_identifier(column.args.get("table"), superset_dialect()),
               resolve_identifier(column.this, superset_dialect()))
        if key not in names:
            raise ValueError(f"the dataset has no name for {column.sql(dialect=superset_dialect())}")
        name = names[key]
        column.replace(exp.Column(this=column.this.copy() if name == key[1] else _identifier(name)))
    return tree.expressions[0].sql(dialect=superset_dialect())
