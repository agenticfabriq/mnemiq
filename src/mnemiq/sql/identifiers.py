"""How the ENGINE resolves an identifier -- the one place that question is answered.

M55. Every resolver in this package has to decide whether two spellings name one object, and each
one that answered independently answered differently. `guard.py` had it wrong twice in consecutive
commits; `scope.py` ignored quoting entirely, which let a quoted CTE hide a base table from the
grant check (M79); `qualify.py` folds case and says so, naming this as the fix; `views.py` widens a
set of spellings and is not yet here.

Three rules, and a dialect needs all of them:

  * WHICH WAY unquoted names fold. Oracle folds them UP where the others fold them down, so one
    hardcoded direction is wrong for one of them -- and wrong in the direction that VOUCHES, since
    a name that matches too widely lets a local alias stand in for a real table.
  * HOW FAR the fold reaches. DuckDB, SQLite and Postgres fold ASCII letters only: measured,
    with a table `ä` and a CTE `Ä`, `... FROM ä` read the table on SQLite 3.50, DuckDB and
    Postgres 16 (UTF8). Python's `str.lower` folds every letter -- and the Kelvin sign to `k` --
    so it merged two objects those engines keep apart, and a CTE vouched for a table read past the
    view check (Codex's review of #144). Oracle folds each letter UP by its one-character
    uppercase: `ä` is stored `Ä`, and `orderſ` reads `ORDERS`; but `straße` is stored `STRAßE`,
    where Python's `str.upper` gives `STRASSE`, one object with `strasse` (measured, Oracle Free).
    The direction of a comparison decides how far it may fold: one that GRANTS or VOUCHES folds no
    wider than the engine -- this resolver, `AccessPolicy.names`, the view check's known objects
    and the facts grant check do (M154). Comparisons that only RESTRICT still fold every letter,
    which applies a filter, mask or denial to more: `qualify.names_one_object`'s default,
    `views.py`'s mention side, the write path's CTE match. The read guard's CTE-name set folds
    every letter too, with no failure reproduced. The normalisation design (M55) owns the rest.
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

import string
from collections.abc import Callable
from dataclasses import dataclass

from sqlglot import exp


@dataclass(frozen=True)
class IdentifierRules:
    fold: Callable[[str], str]
    quoting_preserves_case: bool
    # A character whose unquoted fold on the engine is not measured to be `fold`'s.
    unmeasured: Callable[[str], bool] = lambda c: False


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def ascii_lower(text: str) -> str:
    """Fold ASCII letters only, as DuckDB, SQLite and Postgres fold an identifier."""
    return text.translate(_ASCII_LOWER)


def _oracle_upper(text: str) -> str:
    """Fold as Oracle does: each letter to its uppercase where that is ONE character, and a letter
    whose uppercase is longer -- `ß`, whose is `SS` -- kept as it is. Letters only: a circled
    letter, a small Roman numeral and the combining ypogegrammeni (U+0345) have a one-character
    uppercase too, and no measurement says Oracle folds them, so they are kept -- keeping only
    makes two spellings look like two.

    Swept on Oracle Free (AL32UTF8), each letter as an unquoted alias whose folded name the driver
    reports: of the 5,781 letters below U+3000 Oracle accepts, and all 898 cased letters above it,
    none is folded here where Oracle keeps it -- the direction that would vouch. 27 Greek letters
    with a subscript iota (U+1F80 on) Oracle folds to a single letter and this keeps, since
    Python's uppercase of them is two characters; that only makes two spellings look like two.
    The sweep is `test_live_oracle_sweep_never_folds_a_letter_oracle_keeps`, and a table name
    folds as an alias does (`test_live_oracle_folds_table_names_and_aliases_alike`).
    """
    return "".join(u if c.isalpha() and len(u := c.upper()) == 1 else c for c in text)


def _oracle_unmeasured(c: str) -> bool:
    """A character `_oracle_upper` keeps although it has an uppercase, and whose fold on Oracle
    the sweep did not show to be kept: a Greek letter from U+1F80 on, where the 27 with a
    subscript iota that Oracle folds to one letter are, and a non-letter (a circled letter, a
    small Roman numeral, U+0345), which the sweep did not cover. The other letters it keeps, `ß`
    among them, Oracle was measured to keep."""
    return (c.upper() != c and _oracle_upper(c) == c
            and (not c.isalpha() or 0x1F80 <= ord(c) <= 0x1FFF))


_RULES: dict[str, IdentifierRules] = {
    "postgres": IdentifierRules(ascii_lower, True),
    "oracle": IdentifierRules(_oracle_upper, True, _oracle_unmeasured),
    "duckdb": IdentifierRules(ascii_lower, False),
    "sqlite": IdentifierRules(ascii_lower, False),
}

# An unknown dialect gets the STRICTEST reading, not a guess. Treating quoting as significant, and
# folding ASCII alone, can only make two spellings look like two objects, and this package's
# consumers all fail closed on that -- an extra table node reaches the grant check rather than
# slipping past it.
_UNKNOWN = IdentifierRules(ascii_lower, True)


def rules_for(dialect: str | None) -> IdentifierRules:
    return _RULES.get((dialect or "").lower(), _UNKNOWN)


def resolve(text: str, *, quoted: bool, dialect: str | None) -> str:
    """What this identifier resolves to in `dialect`. Compare these, never the text as typed."""
    rules = rules_for(dialect)
    return text if (quoted and rules.quoting_preserves_case) else rules.fold(text)


def resolve_exactly(text: str, *, quoted: bool, dialect: str | None) -> str | None:
    """`resolve`, or None where an unquoted `text` holds a character the engine may fold another
    way. `resolve` keeps such a character, which only makes two spellings look like two; a check
    that APPROVES a name because it reads as written needs the engine's own reading, and has none
    for these (M223)."""
    rules = rules_for(dialect)
    if not (quoted and rules.quoting_preserves_case) and any(map(rules.unmeasured, text)):
        return None
    return resolve(text, quoted=quoted, dialect=dialect)


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

    As an UNQUOTED identifier, which is right only where the engine never tells names apart by
    case: DuckDB and SQLite keep `Secret` as created and read `secret` as it, and `check_access`
    folds a stored column name this way there (M170). Where quoting keeps case (Postgres, Oracle)
    a mixed-case stored name belongs to a column created with quotes and its fold is a DIFFERENT
    name -- `"Amount"` beside `amount` -- so this must not be applied to such a catalog's names
    without knowing which were quoted (the review gate's case on M170).

    The rest of the plan-versus-snapshot comparison runs on as-typed text: `visible` and the
    policy maps are keyed by stored names, and each check looks a table up by the name as
    written. That is sound only where the engine reads a name as written, so `check_access`
    decides a table reference only there and otherwise says how to write it
    (`authz_guard.not_read_as_written`, M223). This paragraph once said the as-typed comparison
    could only refuse; on Oracle, with a quoted `sales` granted beside SALES, it approved the
    unquoted spelling, which reads SALES. Column names on Postgres and Oracle are still compared
    as typed, which refuses `id` for a stored `ID` (the register's M170 remainder).
    """
    return resolve(name, quoted=False, dialect=dialect)
