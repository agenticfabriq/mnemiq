"""On Oracle, a statement is sent to Superset only as Superset would run it: read from Superset's own dry run.

Superset's `execute_sql` does not run the text it is sent. It reads the statement with its own SQL parser, puts each
table a row rule applies to for the caller in a subquery carrying the rule, puts the call's `limit` in place of the
statement's own, and runs its own rendering of the result (`SQLExecutor._prepare_sql` and `execute` in
`superset/sql/execution/executor.py`). A dry run (`dry_run: true`) answers that rendering, `statements[].executed_sql`,
without running it. `SupersetSqlAdapter` dry-runs each statement it sends on Oracle with the options the real call
carries, asks the extension's `tables_read` which tables Superset's own access check reads in the statement and its
`table_rules` whether a rule applies to the caller on each table the statement reads, and sends the statement only
when `check_tables_read` finds every table mnemiq reads among the tables Superset reads, and `check_executed` finds
that what Superset would run is the statement mnemiq decided:

- read with mnemiq's own parser, with each of Superset's wraps of a table put back as the table, the executed
  statement reads node for node as the statement sent; the top-level row limit is excepted where the call asks
  Superset for one, and must then be a plain count of at most the rows asked;
- every occurrence of every table mnemiq reads is wrapped where a rule applies to it for the caller, and where mnemiq
  cannot tell whether one does;
- every column qualifier of the executed statement names one table or subquery of its query or of a query around it,
  and each of the statement's own names the same source there as it does in the statement sent.

A wrap is the derived table Superset's `RLSAsSubqueryTransformer` (`superset/sql/parse.py`) writes:
`(SELECT * FROM <table> WHERE <rule>) <alias>`, the table as it was but for its alias, which moves to the subquery,
or, for a table with none, its name quoted as written. Its rule is Superset's and is never read here, nor quoted in a
refusal.

No other difference is accepted: a statement Superset would render otherwise in any node is refused, and only a query
is checked, never a text mnemiq's parser keeps as a command it does not model.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.sql.identifiers import resolve_exactly
from mnemiq.sql.scope import base_tables
from mnemiq.superset.names import RefusedStatement

__all__ = ["check_executed", "check_tables_read", "executed_sql", "physical_reads", "rule_key"]

# The longest piece of a statement a refusal quotes.
_QUOTED = 160
# Nodes too small to say where a statement was read otherwise: a refusal quotes the node around one instead.
_LEAVES = (exp.Identifier, exp.Literal, exp.Var, exp.TableAlias, exp.DataType, exp.Star, exp.Null, exp.Boolean)

_NOT_SHOWN = ("Superset's dry run of the statement did not show the one statement it would run, so mnemiq did not send "
              "it; send one SELECT")
_UNREAD = ("Superset would run a different statement from this one, which mnemiq could not read as one query, so it was "
           "not sent; write the statement another way")
_UNSENT = "mnemiq could not read the statement as one query, so it was not sent; send one SELECT"
_UNTOLD = ("mnemiq sends a statement to Superset only when Superset reads each table mnemiq reads in it, and mnemiq could "
           "not tell which tables Superset reads in this one, so it was not sent; send one plain SELECT")


def rule_key(catalog: str, schema: str, table: str) -> str:
    """The key `check_executed` reads a table's rule by: its catalog, schema and name as a statement names them, the
    parts it does not name left out, written as Superset writes a table it looks rules up for (`Table.__str__` in
    `superset/sql/parse.py`)."""
    return ".".join(urllib.parse.quote(part, safe="").replace(".", "%2E") for part in (catalog, schema, table) if part)


def physical_reads(sent: str, *, dialect: str) -> dict[str, tuple[str, str, str]]:
    """Each table mnemiq reads in `sent` (`base_tables`, a CTE's reference left out), by its `rule_key`: the catalog,
    schema and name the statement names it by, "" for a part it does not name. Raises `RefusedStatement` for a text
    that is not one statement."""
    return {rule_key(table.catalog, table.db, table.name): (table.catalog, table.db, table.name)
            for table in base_tables(_statement(sent, dialect, unread=_UNSENT), dialect)}


def executed_sql(payload: Mapping) -> str:
    """The statement a dry run's answer says Superset would run: exactly one, with its `executed_sql`. Raises
    `RefusedStatement` for any other answer."""
    statements = payload.get("statements")
    if payload.get("_response_truncated") or not isinstance(statements, list) or len(statements) != 1:
        raise RefusedStatement(_NOT_SHOWN)
    [statement] = statements
    executed = statement.get("executed_sql") if isinstance(statement, dict) else None
    if not isinstance(executed, str) or not executed.strip():
        raise RefusedStatement(_NOT_SHOWN)
    return executed


def check_tables_read(sent: str, read: frozenset[tuple[str | None, str | None, str]] | None, *, schema: str,
                      dialect: str) -> None:
    """Return when every table mnemiq reads in `sent` (`base_tables`, a CTE's reference left out) is among `read`, the
    tables Superset's own access check reads in it, as the extension's `tables_read` answers them: each `(catalog,
    schema, table)`, qualified as Superset qualifies a table it applies row rules to. A table mnemiq reads matches by
    its name and the schema that qualifies it, `schema` (the session's, in Superset's spelling) where it names none,
    and, where it names one, its catalog; Superset qualifies an unqualified table by the database's own catalog. Raise
    `RefusedStatement` for a table that is not there, and where `read` is None: mnemiq cannot tell."""
    if read is None:
        raise RefusedStatement(_UNTOLD)
    for table in base_tables(_statement(sent, dialect, unread=_UNSENT), dialect):
        at = table.db or schema
        if not any(name == table.name and in_schema == at and (not table.catalog or catalog == table.catalog)
                   for catalog, in_schema, name in read):
            named = _named(table, dialect)
            raise RefusedStatement(f"mnemiq sends a statement to Superset only when Superset reads each table mnemiq "
                                   f"reads in it, which mnemiq could not confirm for {named}, so it was not sent; "
                                   f"read {named} in a plain FROM or JOIN, under an alias of its own")


def check_executed(sent: str, executed: str, *, rules: Mapping[str, bool | None], dialect: str,
                   limit: int | None = None) -> None:
    """Return when `executed`, the statement Superset would run for `sent`, is `sent` as mnemiq read it, each table
    mnemiq reads under its row rule where one may apply; raise `RefusedStatement` saying why not, in words its writer
    can act on. `rules` says, by `rule_key`, whether a rule applies to the caller on each table mnemiq reads: True,
    False, or None where mnemiq cannot tell; a table it leaves out is one mnemiq cannot tell of. `limit` is the row
    limit the call asks Superset for, if any. Both statements are read in `dialect`."""
    decided = _statement(sent, dialect, unread=_UNSENT)
    ran = _statement(executed, dialect, unread=_UNREAD)
    reads = base_tables(decided, dialect)
    walk = _Walk({id(table) for table in reads}, limit)
    if not walk.alike(decided, ran, root=True):
        raise RefusedStatement(_different(walk.at, dialect))
    if limit is not None and not _limited(ran.args.get("limit"), limit):
        raise RefusedStatement(f"Superset would run this statement with another row limit than the {limit:,} rows "
                               "asked, so it was not sent; state the row limit as FETCH FIRST n ROWS ONLY")
    for table in reads:
        told = rules.get(rule_key(table.catalog, table.db, table.name))
        if told is not False and id(table) not in walk.wrapped:
            raise RefusedStatement(_unruled(_named(table, dialect), told))
    _qualifiers(walk, ran, dialect)


def _unruled(named: str, told: bool | None) -> str:
    if told is None:
        return (f"mnemiq sends a statement to Superset only when Superset would apply each row rule of yours to each "
                f"table it reads, and mnemiq could not tell whether one applies to {named}, so it was not sent")
    return (f"mnemiq sends a statement to Superset only when Superset would apply each row rule of yours to each table "
            f"it reads, which mnemiq could not confirm for {named}, so it was not sent; read {named} in a plain FROM "
            "or JOIN, under an alias of its own")


def _statement(text: str, dialect: str, *, unread: str) -> exp.Expression:
    try:
        parsed = [tree for tree in sqlglot.parse(text, read=dialect) if tree is not None]
    except SqlglotError:
        raise RefusedStatement(unread) from None
    # Only a query is checked: a text mnemiq's parser keeps as a command it does not model (`exp.Command`: an EXPLAIN,
    # a CALL, a SHOW) reads no table here, so nothing about it could be confirmed.
    if len(parsed) != 1 or not isinstance(parsed[0], exp.Query) or parsed[0].find(exp.Command) is not None:
        raise RefusedStatement(unread)
    # A table's name part sqlglot keeps as text, no name (the empty middle of `shop..orders`), names nothing here.
    if any(table.args.get(part) is not None and not isinstance(table.args[part], exp.Expression)
           for table in parsed[0].find_all(exp.Table) for part in ("this", "db", "catalog")):
        raise RefusedStatement(unread)
    return parsed[0]


class _Walk:
    """The statement sent and the one Superset would run, walked together: `alike` holds them to be one statement,
    each table of `reads` in the second either as it was or under a wrap (`wrapped`, by the table's id). `placed` maps
    each node of the first, by id, to its counterpart in the second (a wrapped table's is its wrap), `columns` pairs
    their columns, and `at` is the node of the first where they were first read apart."""

    def __init__(self, reads: set[int], limit: int | None) -> None:
        self.reads, self.limit = reads, limit
        self.wrapped: set[int] = set()
        self.placed: dict[int, exp.Expression] = {}
        self.columns: list[tuple[exp.Column, exp.Column]] = []
        self.at: exp.Expression | None = None

    def alike(self, sent, ran, *, root: bool = False) -> bool:
        if isinstance(sent, exp.Table) and id(sent) in self.reads and _wraps(ran, sent):
            self.placed[id(sent)] = ran
            self.wrapped.add(id(sent))
            return True
        if isinstance(sent, exp.Expression) or isinstance(ran, exp.Expression):
            if type(sent) is not type(ran):
                self._read_apart(sent)
                return False
            self.placed[id(sent)] = ran
            if isinstance(sent, exp.Column):
                self.columns.append((sent, ran))
            keys = set(sent.args) | set(ran.args)
            if root and self.limit is not None:
                keys.discard("limit")  # Superset's own, in place of the statement's (`_limited`)
            for key in sorted(keys):
                if not self.alike(sent.args.get(key), ran.args.get(key)):
                    self._read_apart(sent)
                    return False
            return True
        if isinstance(sent, list) and isinstance(ran, list):
            return len(sent) == len(ran) and all(self.alike(one, other) for one, other in zip(sent, ran, strict=True))
        return sent == ran

    def _read_apart(self, node) -> None:
        if self.at is None and isinstance(node, exp.Expression):
            self.at = node


def _wraps(ran, table: exp.Table) -> bool:
    """Whether `ran` is Superset's wrap of `table`: `(SELECT * FROM <table> WHERE <rule>) <alias>`, the table as it
    is but for its alias, which the subquery takes, or for a table with none, its name quoted (`"orders"`)."""
    if not isinstance(ran, exp.Subquery) or _given(ran) - {"this", "alias"}:
        return False
    inner = ran.this
    if not isinstance(inner, exp.Select) or _given(inner) != {"expressions", "from_", "where"}:
        return False
    star, source = inner.expressions, inner.args["from_"]
    if len(star) != 1 or not isinstance(star[0], exp.Star) or _given(star[0]) or _given(source) != {"this"}:
        return False
    bare = table.copy()
    bare.set("alias", None)
    if not isinstance(source.this, exp.Table) or not _same(source.this, bare):
        return False
    alias = table.args.get("alias") or exp.TableAlias(this=exp.Identifier(this=table.name, quoted=True))
    return _same(ran.args.get("alias"), alias)


def _given(node: exp.Expression) -> set[str]:
    """The arguments `node` carries a value in."""
    return {key for key, value in node.args.items() if value is not None and value is not False and value != []}


def _same(a, b) -> bool:
    """Whether two parses are one node: of one type, each argument equal, a name in its exact case and quoting."""
    if isinstance(a, exp.Expression) or isinstance(b, exp.Expression):
        return type(a) is type(b) and all(_same(a.args.get(key), b.args.get(key)) for key in set(a.args) | set(b.args))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(_same, a, b))
    return a == b


def _limited(written, limit: int) -> bool:
    """Whether `written`, the executed statement's top-level row limit, is a plain count of at most `limit` rows:
    Oracle's FETCH FIRST n ROWS ONLY, or a LIMIT n, neither a share of the rows nor keeping its ties."""
    if isinstance(written, exp.Fetch):
        options = written.args.get("limit_options")
        if options is not None and (options.args.get("percent") or options.args.get("with_ties")):
            return False
        count = written.args.get("count")
    elif isinstance(written, exp.Limit) and not written.args.get("offset"):
        count = written.expression
    else:
        return False
    return (isinstance(count, exp.Literal) and not count.is_string and count.this.isdigit()
            and 0 < int(count.this) <= limit)


def _qualifiers(walk: _Walk, ran: exp.Expression, dialect: str) -> None:
    """Raise `RefusedStatement` where a column qualifier of `ran` names no one source of its query or of a query
    around it, or a column of the statement sent is qualified by another source in `ran` than in the statement."""
    sent_of = {id(theirs): mine for mine, theirs in walk.columns}
    for column in ran.find_all(exp.Column):
        if column.args.get("table") is not None and _source_of(column, dialect) is None:
            mine = sent_of.get(id(column))
            named = f" {_quoted(mine, dialect)}" if mine is not None else ""
            raise RefusedStatement(
                f"Superset would run this statement with the column{named} qualified by a name that is no one table "
                "or subquery of its query, so it was not sent; alias each table and subquery, and qualify each column "
                "by its alias")
    for mine, theirs in walk.columns:
        if mine.args.get("table") is None:
            continue
        source = _source_of(mine, dialect)
        if source is not None and walk.placed.get(id(source)) is not _source_of(theirs, dialect):
            raise RefusedStatement(
                f"Superset would run this statement with the column {_quoted(mine, dialect)} qualified by another "
                "table or subquery than the statement reads it from, so it was not sent; alias each table and "
                "subquery, and qualify each column by its alias")


def _source_of(column: exp.Column, dialect: str) -> exp.Expression | None:
    """The source `column`'s qualifier names: among the sources of the nearest query around it that has one by that
    name, as `dialect` reads names; None when no query has one, or the nearest has two."""
    name = _read(column.args.get("table"), dialect)
    if name is None:
        return None
    node = column.parent
    while node is not None:
        if isinstance(node, exp.Select):
            found = [source for source in _sources(node) if _read(_exposed(source), dialect) == name]
            if found:
                return found[0] if len(found) == 1 else None
        node = node.parent
    return None


def _sources(select: exp.Select) -> list[exp.Expression]:
    """The sources of `select`'s FROM and its joins, a table in parentheses standing for the table inside."""
    found = [select.args["from_"].this] if select.args.get("from_") is not None else []
    found += [join.this for join in select.args.get("joins") or []]
    out = []
    for source in found:
        while isinstance(source, exp.Subquery) and source.args.get("alias") is None \
                and isinstance(source.this, (exp.Table, exp.Subquery)):
            source = source.this
        out.append(source)
    return out


def _exposed(source: exp.Expression) -> exp.Expression | None:
    """The identifier a source goes by: its alias, else a table's name."""
    alias = source.args.get("alias")
    if isinstance(alias, exp.TableAlias) and alias.this is not None:
        return alias.this
    return source.this if isinstance(source, exp.Table) else None


def _read(identifier, dialect: str) -> str | None:
    """The name `dialect` reads `identifier` as, or None where it is no identifier or its reading is not known
    exactly (`resolve_exactly`)."""
    if not isinstance(identifier, exp.Identifier):
        return None
    return resolve_exactly(identifier.name, quoted=bool(identifier.args.get("quoted")), dialect=dialect)


def _different(at: exp.Expression | None, dialect: str) -> str:
    if at is None:
        return "Superset would run a different statement from this one, so it was not sent; write it another way"
    return (f"Superset would run a different statement from this one: it reads {_quoted(_around(at), dialect)} "
            "otherwise than mnemiq does, so the statement was not sent; write that part of it another way")


def _around(node: exp.Expression) -> exp.Expression:
    """`node`, or the nearest node around it that is more than a name, a literal or a type."""
    while isinstance(node, _LEAVES) and node.parent is not None:
        node = node.parent
    return node


def _quoted(node: exp.Expression, dialect: str) -> str:
    text = " ".join(node.sql(dialect=dialect).split())
    return text if len(text) <= _QUOTED else text[:_QUOTED - 3] + "..."


def _named(table: exp.Table, dialect: str) -> str:
    """`table`'s name, its schema and catalog with it, as the statement writes them."""
    return exp.Table(**{part: table.args[part].copy() for part in ("this", "db", "catalog")
                        if isinstance(table.args.get(part), exp.Expression)}).sql(dialect=dialect)
