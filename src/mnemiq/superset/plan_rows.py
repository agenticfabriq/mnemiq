"""The rows of a dashboard plan whose columns are named with their tables (`order_items.revenue`, `customers.region`):
each measure's base table, the chain of LEFT JOINs from that base to every other table a part needs, and the name the
joined dataset gives each column. Nothing here reads a source or Superset; the planner reads a draft through it.

A measure's base is the table whose rows it adds up: the table of the columns its aggregates add up (`base_of`). The
other tables a part needs are joined to the base along links, each an equality of one column of each table: the links
the model states, and for a pair of tables it left unlinked, the snapshot's relationships (`edges`). A table is joined
by a link only where the link's columns on that table, its far side, hold each value once, so each of the base's rows
meets at most one of its rows, and LEFT JOIN keeps every base row: the measure's value is the same whichever tables a
chart joins. A far side known to repeat is never joined; one not known either way may be (`far_side`), and the plan
check measures it before the plan is accepted. Each needed table is reached by its shortest chain of such links
(`chain`); two chains of one length, or two links between the same two tables, are refused as ambiguous, since each
gives the measure different rows to read.

Every identifier is read as `SUPERSET_DIALECT` reads it (`resolve_identifier`): an unquoted name folds as the dialect
folds it, and a quoted one is the name as written. Every identifier built here is quoted, so it reads back as the
stored name it was built from; a column whose name the joined dataset keeps is left as the model wrote it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.contract.semantic import Relationship
from mnemiq.sql.fanout_check import value_terms
from mnemiq.sql.identifiers import resolve_identifier
from mnemiq.superset.settings import SUPERSET_DIALECT

# Whether a table's column holds each value once, by (table, column): enrichment's key facts. A column absent is
# unknown.
Unique = Mapping[tuple[str, str], bool]


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
    """`expression` parsed as the one column of a SELECT, or None when it is not one SQL expression."""
    try:
        tree = sqlglot.parse_one(f"SELECT {expression}", read=SUPERSET_DIALECT)
    except SqlglotError:
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
        written = column.sql(dialect=SUPERSET_DIALECT)
        if isinstance(column.this, exp.Star):
            return f"{written} is not one column: name each column with its table, as <table>.<column>"
        if column.args.get("db") or column.args.get("catalog"):
            return f"{written} names a schema: name each column with its table alone, as <table>.<column>"
        if not column.table:
            return f"{column.name!r} is named without its table: write it as <table>.{written}"
        named[(resolve_identifier(column.args["table"], SUPERSET_DIALECT),
               resolve_identifier(column.this, SUPERSET_DIALECT))] = None
    return list(named)


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
            tables.update(resolve_identifier(column.args["table"], SUPERSET_DIALECT) for column in term
                          if column.table)
    return tables


def base_of(expression: str, table: str | None) -> str | Refusal:
    """The table whose rows a written measure adds up: the one table of the columns its aggregates add up; where they
    add up no column (`COUNT(*)`), `table`, the table the measure names. `table` decides nothing for an expression whose
    aggregates name a column. Refused when the expression cannot be read (`qualified`), when its aggregates add up
    columns of two tables, or when they add up none and it names no table."""
    named = qualified(expression)
    if isinstance(named, str):
        return Refusal(named)
    tables = sorted(aggregated(expression))
    if len(tables) > 1:
        return Refusal(f"it adds up columns of {_listed(tables)}, and a measure adds up the rows of one table: plan "
                       "one measure for each table")
    if tables:
        return tables[0]
    if table is not None:
        return table
    return Refusal("its aggregate names no column, so it does not say whose rows it adds up: give the measure its "
                   "table, or aggregate one of that table's columns, as COUNT(<table>.<column>)")


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
          unique: Unique) -> tuple[Step, ...] | Refusal:
    """The LEFT JOINs that bring each of `needed` to `base`'s rows, in join order: for each needed table, the shortest
    chain of links from `base` over which each joined table's far side is not known to repeat (`far_side`). Refused,
    naming them, when a needed table is reached by no such chain; when one is reached by two chains of one length, or
    by a step whose two tables two such links join (an employee's department and the department an employee manages),
    since either gives the base's rows other rows to read."""
    adjacent: dict[str, list[tuple[str, frozenset[str]]]] = {}
    for pair in edges:
        for near, far in (sorted(pair), sorted(pair, reverse=True)):
            adjacent.setdefault(near, []).append((far, pair))
    # A breadth-first walk from the base: each table's depth, every table one shorter that reaches it, and the links
    # each such step may join by.
    depth, parents, order = {base: 0}, {base: []}, [base]
    usable: dict[tuple[str, str], list[tuple[Link, ...]]] = {}
    blocked: list[tuple[Link, ...]] = []
    frontier = [base]
    while frontier:
        reached: list[str] = []
        for near in frontier:
            for far, pair in sorted(adjacent.get(near, ())):
                if far in depth and depth[far] <= depth[near]:
                    continue
                ways = _ways(edges[pair], near)
                joinable = [on for on in ways if far_side(far, [link.right[1] for link in on], unique) is not False]
                if not joinable:
                    blocked.extend(ways)
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
                  "columns on each table it joins hold each value once")
        for on in blocked:
            if on[0].right[0] in missing:
                columns = ", ".join(dict.fromkeys(f"{link.right[0]}.{link.right[1]}" for link in on))
                reason += (f"; the link {' and '.join(map(_link_text, on))} would repeat rows of {on[0].left[0]!r}, "
                           f"since {columns} holds a value more than once")
        return Refusal(reason)

    def path_to(table: str) -> list[str]:
        """`table`'s chain from the base, through each table's first parent."""
        path = [table]
        while parents[path[-1]]:
            path.append(parents[path[-1]][0])
        return path[::-1]

    def path_text(path: Sequence[str]) -> str:
        return " then ".join(_link_text(link) for near, far in zip(path, path[1:]) for link in usable[(near, far)][0])

    joined: set[str] = set()
    for table in wanted:
        walked = [table]
        while parents[walked[-1]]:
            node = walked[-1]
            if len(parents[node]) > 1:
                rest = walked[::-1][1:]
                first, second = (path_to(parent) + [node] + rest for parent in parents[node][:2])
                return Refusal(f"{table!r} is reached from the rows of {base!r} two ways of one length, by "
                               f"{path_text(first)}, and by {path_text(second)}; mnemiq joins a table one way only")
            near = parents[node][0]
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
    return tuple(Step(table, usable[(parents[table][0], table)][0])
                 for table in order if table in joined and table != base)


def _identifier(name: str) -> exp.Identifier:
    return exp.to_identifier(name, quoted=True)


def _column(table: str, column: str) -> exp.Column:
    return exp.Column(this=_identifier(column), table=_identifier(table))


def join_text(base: str, steps: Sequence[Step]) -> str:
    """The FROM clause `steps` describe: `base` LEFT JOIN each step's table ON its links' columns being equal, every
    name quoted as `SUPERSET_DIALECT` writes it."""
    text = exp.Table(this=_identifier(base)).sql(dialect=SUPERSET_DIALECT)
    for step in steps:
        on = exp.and_(*(exp.EQ(this=_column(*link.left), expression=_column(*link.right)) for link in step.on))
        text += " " + exp.Join(this=exp.Table(this=_identifier(step.table)), on=on, side="LEFT").sql(
            dialect=SUPERSET_DIALECT)
    return text


def rewrite(expression: str, names: Mapping[tuple[str, str], str]) -> str:
    """`expression` with each (table, column) it names written as the joined dataset's name for it in `names`, without
    a table. A column the dataset names as its table does keeps the identifier written, quoting and all, so the
    measure reads as one written over that table alone (`SUM(order_items.revenue)` is `SUM(revenue)`); a renamed one
    is quoted, so it reads back as the name given. `expression` is one `qualified` reads, and `names` names each of its
    columns; a column it does not name is a ValueError."""
    tree = sqlglot.parse_one(f"SELECT {expression}", read=SUPERSET_DIALECT)
    for column in list(tree.find_all(exp.Column)):
        key = (resolve_identifier(column.args.get("table"), SUPERSET_DIALECT),
               resolve_identifier(column.this, SUPERSET_DIALECT))
        if key not in names:
            raise ValueError(f"the dataset has no name for {column.sql(dialect=SUPERSET_DIALECT)}")
        name = names[key]
        column.replace(exp.Column(this=column.this.copy() if name == key[1] else _identifier(name)))
    return tree.expressions[0].sql(dialect=SUPERSET_DIALECT)
