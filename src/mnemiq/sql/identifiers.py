"""How the ENGINE resolves an identifier -- the one place that question is answered.

M55. Every resolver in this package has to decide whether two spellings name one object, and each
one that answered independently answered differently. `guard.py` had it wrong twice in consecutive
commits; `scope.py` ignored quoting entirely, which let a quoted CTE hide a base table from the
grant check (M79); `qualify.py` folds case and says so, naming this as the fix; `views.py` widens a
set of spellings and is not yet here.

Two rules, and a dialect needs both:

  * WHICH WAY unquoted names fold. Oracle folds them UP where the others fold them down, so one
    hardcoded direction is wrong for one of them -- and wrong in the direction that VOUCHES, since
    a name that matches too widely lets a local alias stand in for a real table.
  * WHETHER quoting makes a name case-SENSITIVE. Postgres and Oracle preserve a quoted name, so
    `"Claim"` and `Claim` are two objects. DuckDB and SQLite fold quoted names too, so they are one.
    Measured on duckdb 1.5.4: `WITH claim AS (...) SELECT * FROM "CLAIM"` reads the CTE.

Getting the second rule wrong in either direction has a cost, which is why it is a flag rather than
an assumption. Too wide and a quoted CTE vouches for an unquoted reference the engine sends to a
base table. Too narrow and `WITH "claim" AS (...) SELECT * FROM claim` -- one object everywhere
that folds down, and the ordinary shape of a model quoting a definition but not its reference -- is
refused.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlglot import exp


@dataclass(frozen=True)
class IdentifierRules:
    fold: Callable[[str], str]
    quoting_preserves_case: bool


_RULES: dict[str, IdentifierRules] = {
    "postgres": IdentifierRules(str.lower, True),
    "oracle": IdentifierRules(str.upper, True),
    "duckdb": IdentifierRules(str.lower, False),
    "sqlite": IdentifierRules(str.lower, False),
}

# An unknown dialect gets the STRICTEST reading, not a guess. Treating quoting as significant can
# only make two spellings look like two objects, and this package's consumers all fail closed on
# that -- an extra table node reaches the grant check rather than slipping past it.
_UNKNOWN = IdentifierRules(str.lower, True)


def rules_for(dialect: str | None) -> IdentifierRules:
    return _RULES.get((dialect or "").lower(), _UNKNOWN)


def resolve(text: str, *, quoted: bool, dialect: str | None) -> str:
    """What this identifier resolves to in `dialect`. Compare these, never the text as typed."""
    rules = rules_for(dialect)
    return text if (quoted and rules.quoting_preserves_case) else rules.fold(text)


def resolve_identifier(node: exp.Expression | None, dialect: str | None) -> str | None:
    """Resolve an `exp.Identifier`, or None when there is nothing to read."""
    while isinstance(node, exp.TableAlias):
        node = node.this
    if not isinstance(node, exp.Identifier):
        return None
    return resolve(node.name, quoted=bool(node.args.get("quoted")), dialect=dialect)


def resolve_name(node: exp.Expression, dialect: str | None) -> str | None:
    """Resolve the name a node is KNOWN BY.

    A CTE is known by the name it DEFINES and a table reference by the name it READS, and those
    live in different places: `alias_or_name` on `claim AS c` is the alias `c`, so reading it for
    both made every aliased reference miss.
    """
    if isinstance(node, exp.CTE):
        return resolve_identifier(node.args.get("alias"), dialect)
    if isinstance(node, exp.Table):
        return resolve_identifier(node.this, dialect)
    return resolve_identifier(node, dialect)


def resolve_stored(name: str, dialect: str | None) -> str:
    """Resolve a name that came from a CATALOG rather than from a query.

    Always as an UNQUOTED identifier: the database has already applied its own folding by the time
    a name reaches its data dictionary, so `CREATE TABLE Claim` is `claim` in Postgres and `CLAIM`
    in Oracle. Re-folding it is what makes a stored name comparable to a resolved reference.
    """
    return resolve(name, quoted=False, dialect=dialect)
