"""Names as Superset shows them and as the database stores them: the one mapping at the Superset boundary.

Superset reads its database through SQLAlchemy, whose Oracle dialect shows a name in its own spelling: a name stored all
upper case that needs no quotes in lower case (`ORDERS` is `orders`), any other as stored (`OrderItems`, `ORDER#`, the
reserved `ORDER`). That is SQLAlchemy's `normalize_name`, and `denormalize_name` reads a name back (both in
`sqlalchemy/engine/default.py`, with the Oracle identifier preparer's `_requires_quotes`, in
`sqlalchemy/dialects/oracle/base.py`). Superset 7.0.0rc2's table list, so a dataset's `table_name`, and its schema
list are in that spelling. A physical dataset's columns are not, unless the dataset is set to normalize them: rc2 names
each as the database stores it (`get_physical_table_metadata` in `superset/connectors/sqla/utils.py` denormalizes it),
and a virtual dataset's as the database returns its query's columns, so a chart names a column as the snapshot stores
it. mnemiq's snapshot, its prompts and every statement it decides keep the database's spelling; a name is mapped only
where it crosses to or from Superset:

- `superset_name`: a stored name as Superset shows it, for a name mnemiq gives Superset to match against its own (the
  table of a dataset the build makes);
- `stored_name`: the stored name Superset reads for a name it shows. That is also what SQLAlchemy's Oracle dialect
  writes the name as in SQL: quoted where `_requires_quotes`, else bare, which Oracle reads in upper case;
- `for_superset`: a statement as mnemiq sends it, each table it reads named as Superset names the table: rc2's access
  check matches a statement's tables to dataset names exactly (`raise_for_access` with `force_dataset_match`), and its
  row rules attach to a table by the same names, a virtual dataset's tables included;
- `from_superset`: SQL Superset holds (a virtual dataset's), each table it reads quoted as the database reads it, the
  only spelling mnemiq's deciders read a table by.

On PostgreSQL Superset shows every name as stored, and each of these is the identity: a statement comes back as it
was given, not printed again.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from mnemiq.sql.identifiers import resolve_exactly
from mnemiq.sql.scope import base_tables
from mnemiq.superset.settings import superset_dialect

# SQLAlchemy 2.0's Oracle `RESERVED_WORDS`, as its identifier preparer holds them, in lower case: a name among them is
# never shown in lower case. Fewer than the words Oracle reserves; SQLAlchemy's list decides how Superset shows a name.
_RESERVED = frozenset(
    "share raw drop between from desc option prior long then default alter is into minus integer number grant "
    "identified all to order on float date having cluster nowait resource any table index for update where check "
    "smallint with delete by asc revoke like size rename nocompress null group values as in view exclusive compress "
    "synonym select insert exists not trigger else create intersect pctfree distinct user connect set mode of unique "
    "varchar2 varchar lock or char decimal union public and start uid comment current level".split())
# The characters SQLAlchemy writes a name with unquoted (its `LEGAL_CHARACTERS`, matched as it matches it), and those
# the Oracle preparer quotes a name for opening with.
_LEGAL = re.compile(r"^[A-Z0-9_$]+$", re.I)
_ILLEGAL_INITIAL = frozenset("0123456789_$")

# The parts of a table reference Superset compares by name: the table, its schema, and a catalog.
_PARTS = ("this", "db", "catalog")


class UnreadableStatement(ValueError):
    """A statement whose tables mnemiq cannot rewrite as one statement it has read: it does not parse as one, or the
    rewrite would not read back as the same statement."""


def _requires_quotes(value: str) -> bool:
    """SQLAlchemy's Oracle `_requires_quotes`: whether it writes `value` quoted."""
    lowered = value.lower()
    return (lowered in _RESERVED or value[0] in _ILLEGAL_INITIAL or not _LEGAL.match(value)
            or lowered != value)


def superset_name(stored: str) -> str:
    """`stored`, a name as the database stores it, as Superset shows it: SQLAlchemy's Oracle `normalize_name` on Oracle,
    the name itself on PostgreSQL."""
    if superset_dialect() != "oracle":
        return stored
    lower, upper = stored.lower(), stored.upper()
    if upper == lower:
        return stored
    if upper == stored and not _requires_quotes(lower):
        return lower
    return stored


def stored_name(shown: str) -> str:
    """The stored name Superset reads for `shown`, a name it shows: SQLAlchemy's Oracle `denormalize_name` on Oracle,
    the name itself on PostgreSQL. A name stored in lower case is shown as itself and reads back in upper case, as
    SQLAlchemy reads it: what Superset reads by it."""
    if superset_dialect() != "oracle":
        return shown
    lower, upper = shown.lower(), shown.upper()
    if upper == lower:
        return shown
    if lower == shown and not _requires_quotes(lower):
        return upper
    return shown


def for_superset(sql: str) -> str:
    """`sql`, a statement mnemiq sends, with each table it reads named as Superset names the table: on Oracle bare and
    in lower case where Superset shows it so (`superset_name`) and Oracle reads that as the same table, else quoted as
    stored. Only the tables it reads change, where they stand; its columns, aliases, CTEs, functions and literals are
    as written. Raises `UnreadableStatement` for a statement mnemiq cannot read as one, which is not to be sent: one of
    its tables in the database's spelling would keep its dataset's row rules from attaching. On PostgreSQL, `sql`."""
    if superset_dialect() != "oracle":
        return sql
    return _rewritten(sql, _supersets_spelling)


def from_superset(sql: str) -> str:
    """`sql`, SQL Superset holds, with each table it reads quoted as the database reads it, as mnemiq's deciders read a
    table; the rest as written. Raises `UnreadableStatement` as `for_superset` does. On PostgreSQL, `sql`."""
    if superset_dialect() != "oracle":
        return sql
    return _rewritten(sql, _quoted)


def _supersets_spelling(stored: str) -> exp.Identifier:
    shown = superset_name(stored)
    if shown != stored and resolve_exactly(shown, quoted=False, dialect="oracle") == stored:
        return exp.to_identifier(shown, quoted=False)
    return _quoted(stored)


def _quoted(stored: str) -> exp.Identifier:
    return exp.to_identifier(stored, quoted=True)


def _rewritten(sql: str, spelled: Callable[[str], exp.Identifier]) -> str:
    """`sql` with each part of each table it reads (`base_tables`) written as `spelled` writes the stored name Oracle
    reads it as, replaced where it stands in the text, so nothing else is printed again. A part whose reading is not
    known exactly (`resolve_exactly`) is left as written. Read back, the text must parse to the statement with those
    parts replaced, or `UnreadableStatement`."""
    try:
        parsed = [tree for tree in sqlglot.parse(sql, read="oracle") if tree is not None]
    except SqlglotError as exc:
        raise UnreadableStatement("the statement does not parse") from exc
    if len(parsed) != 1:
        raise UnreadableStatement("the text holds no statement, or more than one")
    tree = parsed[0]
    spans: list[tuple[int, int, str]] = []
    for table in base_tables(tree, "oracle"):
        for part in _PARTS:
            identifier = table.args.get(part)
            if not isinstance(identifier, exp.Identifier):
                continue
            stored = resolve_exactly(identifier.name, quoted=bool(identifier.args.get("quoted")), dialect="oracle")
            if stored is None:
                continue
            start, end = identifier.meta.get("start"), identifier.meta.get("end")
            if start is None or end is None:
                raise UnreadableStatement("a table's name has no place in the text")
            written = spelled(stored)
            text = written.sql(dialect="oracle")
            if sql[start:end + 1] != text:
                spans.append((start, end + 1, text))
                identifier.replace(written)
    if not spans:
        return sql
    out = sql
    for start, end, text in sorted(spans, reverse=True):
        out = out[:start] + text + out[end:]
    try:
        reread = sqlglot.parse_one(out, read="oracle")
    except SqlglotError as exc:
        raise UnreadableStatement("the rewritten statement does not parse") from exc
    if not _same_tree(reread, tree):
        raise UnreadableStatement("the rewritten statement does not read as the statement with its tables renamed")
    return out


def _same_tree(a, b) -> bool:
    """Whether two parses are one statement: node for node of one type, each argument equal, a name in its exact case
    and quoting. Where a parse puts each node in the text is not compared."""
    if isinstance(a, exp.Expression) or isinstance(b, exp.Expression):
        return type(a) is type(b) and all(_same_tree(a.args.get(key), b.args.get(key))
                                          for key in set(a.args) | set(b.args))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(_same_tree, a, b))
    return a == b
