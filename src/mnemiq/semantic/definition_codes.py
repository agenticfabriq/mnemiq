"""Definitions that name codes their column never holds.

A definition is certified meaning the writer is told to follow. One written for other data (another plant's codes, a
code list since renamed) names codes the source does not hold: the writer filters on them, the value check refuses
each query, and the question is declined because of the definition, with nothing at load to say so.

`stale_codes` reads each definition for the codes it names for an indexed column, and reports the column when it holds
NONE of them. One real value among them clears it: a code list may name codes the data has not seen yet, and a
definition may name the code it says to exclude beside the one to keep. So a report means the definition and the
column share no code at all.

What counts, kept narrow so prose is not read as codes:

- A column is named when its name, holding an underscore or a digit, appears as a whole word (any case): a plain-word
  name such as `status` is also English and is never read from prose.
- A code is a quoted literal, or a bare token of capitals and digits holding both, such as `AB12`, that is not itself
  a column or table name. A quoted literal opens after a non-word character and closes before one, and neither starts
  nor ends with whitespace, so the apostrophe in "lot's" never opens one and swallows the code after it.
- Only a code the definition selects on counts. A mention in an explicit exclusion (`<> 'X'`, `!= 'X'`,
  `NOT IN ('X')`, `NOT (... 'X')`, "is not 'X'", "other than", "except", "excluding", "exclude ... = 'X'" or
  "... IN ('X')", "'X' ... are excluded") is filtered out by a query that follows the definition, which keeps every
  row whether or not the data holds it, so a definition that only excludes is never reported. A nearby negation that
  still selects ("not only 'X' but also", "do not omit ... 'X'", "do not exclude ... 'X'") is not an exclusion, and
  "exclude" reaches only its own predicate. After an SQL operator a code counts only in a form that selects it
  exactly (`=`, `IN (`, `IS NOT DISTINCT FROM`); after any other (`<`, `BETWEEN`, `IS DISTINCT FROM`, a pattern's
  `LIKE` or `ILIKE`, ...) it does not.
- A code counts for a column only when it has the shape of one of the column's values: capitals, small letters and
  digits in the same places. `'from May to June'` is never a code for a column of `AB12`s. It counts for the column
  named last before it in its clause, indexed or not, and for no other; with no column before it, for the one judged
  column that fits, unless several fit or a column that cannot be judged is named too.
- A column is judged only when every column of that name in the snapshot is indexed: a code may be in the one whose
  values were not read, which is unknown, not absent.

The value index already leaves out keys, personal data and columns with too many values, so none of those is judged.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from mnemiq.authz.grants import GrantSet
from mnemiq.contract import Column, Definition, Snapshot
from mnemiq.semantic.measures import read_raw
from mnemiq.semantic.textmatch import similarity

# Opened after a non-word character, closed before one, never starting or ending in whitespace: "'AX' or 'AY'" yields AX
# and AY, not " or ", and "the lot's mode='-A' or '-G'" yields -A and -G, not "s mode=".
_QUOTED = re.compile(r"(?<!\w)'([^'\s](?:[^'\n]{0,62}[^'\s])?)'(?!\w)|(?<!\w)\"([^\"\s](?:[^\"\n]{0,62}[^\"\s])?)\"(?!\w)")
# A clause ends at a full stop (a . before whitespace or the end, never a qualified name's), ; : ?, a blank line, or a !
# that does not begin !=. A single newline inside SQL ends nothing.
_CLAUSE_END = re.compile(r"\.(?=\s|$)|[;:?]|\n\s*\n|!(?!=)")
# An explicit exclusion just before the code, in its clause...
_EXCLUDED_BEFORE = re.compile(
    r"(?:<>|!=|\bnot\s+in\s*\((?:[^()]*,)?|\b(?:is|are|was|were)\s+(?:not|never)|\bnot\s+equals?(?:\s+to)?|\bother\s+than"
    r"|\bexcept(?:\s+for)?|\bexcluding|\bbut\s+not"
    # "exclude ... =" or "exclude ... IN (", not negated, and reaching back over no other quoted code: its own
    # predicate only.
    r"|(?<!\bnot )(?<!\bnever )(?<!n't )\bexclud\w*\b(?:[^.;:?'\"]|\.(?!\s|$))*?(?:=|\bin\s*\())\s*$", re.IGNORECASE)
# ...or just after it. Both read a coordinated list ("'X', 'Y' and 'Z'") as one mention.
_EXCLUDED_AFTER = re.compile(
    r"^\s*\)?\s*(?:rows?\s+)?(?:are|is)\s+(?:excluded|ignored|dropped|removed|left\s+out|filtered\s+out)\b", re.IGNORECASE)
# An SQL operator just before the code. There a code counts only in a form that selects it exactly (=, IN (, IS NOT
# DISTINCT FROM). A LIKE or ILIKE compares by pattern, ILIKE folding case and some engines' LIKE too, so exact
# membership cannot say it matches nothing (`ILIKE 'ab12'` finds AB12), and the finding reaches the writer as a note
# (Codex on the note). Any other (<, >, BETWEEN, IS DISTINCT FROM, NOT LIKE, SIMILAR TO, ...) may keep every row whether or
# not the data holds the code, so the code does not count. Prose before a code is read as prose.
_SQL_BEFORE = re.compile(
    r"(?:<>|!=|<=|>=|[=<>]|\bin\s*\((?:[^()]*,)?|\b(?:i?like|from|between|similar\s+to|regexp|rlike|glob)"
    r"|\b(?:any|all)\s*\()\s*$", re.IGNORECASE)
_SQL_SELECTS = re.compile(
    r"(?:(?<![<>!])=|\bin\s*\((?:[^()]*,)?|\bis\s+not\s+distinct\s+from)\s*$", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
# The words between a column and its code. Where they are SQL, the code counts only after an exact selecting form, and
# the list it heads: a pattern behind a parenthesis or a comment, or a function around the column or the code, may
# match what exact membership calls absent, and the finding reaches the writer as an instruction (Codex's second
# pass on the note). Comments are read as space.
_COMMENT = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)
# A comparison, a pattern word, or a parenthesis that opens on the code or closes on the column (a function around
# either) is SQL; a parenthesis in the words between is prose ("the ingot tables (A, B) belong to plant X").
_SQL_GAP = re.compile(r"[<>=!]|\b(?:i?like|similar\s+to|regexp|rlike|glob|between)\b|\(\s*$|^\s*\)", re.IGNORECASE)
_LISTED = r"(?:(?:'[^']*'|\"[^\"]*\"|[A-Za-z0-9_-]+)\s*(?:,|\bor\b|\band\b)\s*)*"
_EXACT_GAP = re.compile(rf"\s*(?:=|\bin\s*\(|\bis\s+not\s+distinct\s+from)\s*\(?\s*{_LISTED}", re.IGNORECASE)
# A negation opening a parenthesised predicate, as SQL writes one.
_NOT_PAREN = re.compile(r"\bnot\s*\(", re.IGNORECASE)
# What may stand between two codes of one list: a comma, "and", "or", or both.
_LIST_GAP = re.compile(r"\s*(?:,\s*)?(?:(?:and|or)\s+)?", re.IGNORECASE)
_BARE = re.compile(r"(?<![\w'\"-])[A-Z0-9][A-Z0-9_-]*(?![\w'\"-])")
NEAREST = 8  # real values quoted, as many as the value check quotes
# Column types that can hold a code, as the value index reads them (semantic/values.py).
_TEXT_TYPES = ("char", "text", "string", "clob")


@dataclass(frozen=True)
class StaleCodes:
    """A definition that names `codes` for `column`, none of which any column of that name holds."""

    term: str
    column: str
    objects: tuple[str, ...]  # the objects whose column was read
    codes: tuple[str, ...]
    nearest: tuple[str, ...]  # the column's real values most like the first code, best first
    # The text the codes were read from: a note rides only with a definition that still says it (`noted_stale`).
    definition: str = ""


def _shape(text: str) -> str:
    return "".join("A" if c.isupper() else "a" if c.islower() else "9" if c.isdigit() else c for c in text)


def _shows_exact(gap: str) -> bool:
    """Whether `gap`, the words before a code, lets it count: prose, or SQL that selects it exactly (`_EXACT_GAP`)."""
    gap = _COMMENT.sub(" ", gap)
    return _SQL_GAP.search(gap) is None or _EXACT_GAP.fullmatch(gap) is not None


def _identifier_like(name: str) -> bool:
    return "_" in name or any(c.isdigit() for c in name)


def _named_at(column: str, text: str) -> list[int]:
    """Where `text` names `column`, as a whole word in any case."""
    return [m.start() for m in re.finditer(rf"(?<![\w]){re.escape(column)}(?!\w)", text, re.IGNORECASE)]


def _inside_not(before: str) -> bool:
    """Whether `before` ends inside a `NOT (` it opened and has not closed: `NOT (SHIP_MODE IN ('X'))`."""
    for opened in reversed(list(_NOT_PAREN.finditer(before))):
        rest = before[opened.end():]
        if rest.count("(") - rest.count(")") >= 0:
            return True
    return False


def _excluded(text: str, start: int, end: int) -> bool:
    """Whether the mention at `text[start:end]` does not select its code: an explicit exclusion within its own clause,
    or an SQL operator before it that is not a selecting form."""
    # Whitespace collapsed, so "NOT  LIKE", "NOT\nLIKE" and "not  exclude" read as their one-space forms.
    before = _SPACE.sub(" ", _CLAUSE_END.split(text[:start])[-1])
    after = _SPACE.sub(" ", _CLAUSE_END.split(text[end:], maxsplit=1)[0])
    sql = _SQL_BEFORE.search(before) is not None and _SQL_SELECTS.search(before) is None
    return (_EXCLUDED_BEFORE.search(before) is not None or _EXCLUDED_AFTER.search(after) is not None
            or _inside_not(before) or sql)


def _codes(text: str, names: set[str]) -> list[tuple[str, int]]:
    """The codes the text selects on, each with where it is: every quoted or bare mention outside an exclusion.

    Consecutive mentions joined only by a comma, "and" or "or" are one list, read as one mention: "'X' and 'Y' are
    excluded" excludes both, and "exclude ... = 'X' or 'Y'" both. A gap that ends a clause (a blank line) joins
    nothing, so a list never carries a code into the next clause's exclusion."""
    quoted = [(m.group(1) or m.group(2), m.start(), m.end()) for m in _QUOTED.finditer(text)]
    bare = [(m.group(0), m.start(), m.end()) for m in _BARE.finditer(text)
            if m.group(0).lower() not in names and any(c.isalpha() for c in m.group(0))
            and any(c.isdigit() for c in m.group(0))]
    lists: list[list[tuple[str, int, int]]] = []
    for mention in sorted(quoted + bare, key=lambda m: m[1]):
        gap = text[lists[-1][-1][2]:mention[1]] if lists else ""
        if lists and _LIST_GAP.fullmatch(gap) and _CLAUSE_END.search(gap) is None:
            lists[-1].append(mention)
        else:
            lists.append([mention])
    return [(code, start) for group in lists if not _excluded(text, group[0][1], group[-1][2])
            for code, start, _ in group]


def _judged(snapshot: Snapshot, values) -> dict[str, tuple[str, tuple[str, ...], list[str]]]:
    """Each column name that can be judged, lowercased: as the snapshot spells it, the objects that hold it, and every
    value they hold. Only identifier-like names, and only when every column of that name is indexed."""
    read = getattr(values, "values", None)
    if read is None:
        return {}
    by_name: dict[str, list[tuple[str, str]]] = {}
    for col in snapshot.columns:
        by_name.setdefault(col.name.lower(), []).append((col.object_id, col.name))
    judged: dict[str, tuple[str, tuple[str, ...], list[str]]] = {}
    for key, cols in by_name.items():
        if not _identifier_like(key) or not all(values.has(obj, name) for obj, name in cols):
            continue
        held = sorted({v for obj, name in cols for v in read(obj, name)})
        if held:
            judged[key] = (cols[0][1], tuple(sorted(obj for obj, _ in cols)), held)
    return judged


def _unjudged(snapshot: Snapshot, judged: dict) -> set[str]:
    """Column names, lowercased, that may hold a code but cannot be judged: text or untyped columns the index does not
    hold whole, plain-word names (`status`) included. A code in such a column's predicate is that column's, and
    unknown."""
    out = set()
    for col in snapshot.columns:
        key = col.name.lower()
        typed = (col.data_type or "").lower()
        if key not in judged and (not typed or any(t in typed for t in _TEXT_TYPES)):
            out.add(key)
    return out


def _attributed(text: str, judged: dict, names: set[str], unjudged: set[str]) -> dict[str, list[str]]:
    """The codes `text` selects on, each credited to the judged column it is for, by column key.

    A code is for the column named last before it in its own clause, judged or not: credited when that column is judged
    and the code has the shape of its values, and skipped otherwise, since it is that column's and not another's. A
    code with no column before it in its clause is credited to the one judged column the text names whose values share
    its shape, and skipped when several do or when the text names an identifier-like column that cannot be judged:
    which column it is for cannot be told, and a guess could report a column it was never about. A plain-word name
    (`status`) takes the codes of its own predicate but stops no other, since the same word in prose names nothing."""
    judged_at = {key: at for key, (column, _, _) in judged.items() if (at := _named_at(column, text))}
    other_at = {key: at for key in unjudged if (at := _named_at(key, text))}
    shapes = {key: {_shape(v) for v in judged[key][2]} for key in judged_at}
    out: dict[str, list[str]] = {}
    for code, start in _codes(text, names):
        clause = start - len(_CLAUSE_END.split(text[:start])[-1])
        before = [(at, key) for key, ats in (judged_at | other_at).items() for at in ats if clause <= at < start]
        if before:
            at, key = max(before)
            if (key in judged_at and _shape(code) in shapes[key]
                    and _shows_exact(text[at + len(judged[key][0]):start])):
                out.setdefault(key, []).append(code)
            continue
        fits = [key for key in judged_at if _shape(code) in shapes[key]]
        if len(fits) == 1 and not any(_identifier_like(key) for key in other_at) and _shows_exact(text[clause:start]):
            out.setdefault(fits[0], []).append(code)
    return out


def stale_codes(snapshot: Snapshot, values, definitions: Sequence[Definition] | None = None) -> list[StaleCodes]:
    """Each definition and indexed column where the definition names codes the column holds none of.

    `values` is the value index (`ValueIndex`), read through `has` and `values`; an index that holds nothing
    (`NoValueIndex`) judges nothing. `definitions` are the ones read, the snapshot's when None."""
    judged = _judged(snapshot, values)
    names = {col.name.lower() for col in snapshot.columns} | {col.object_id.lower() for col in snapshot.columns}
    unjudged = _unjudged(snapshot, judged)
    found: list[StaleCodes] = []
    for definition in snapshot.definitions if definitions is None else definitions:
        for key, codes in sorted(_attributed(definition.definition, judged, names, unjudged).items()):
            column, objects, held = judged[key]
            named = sorted(set(codes))
            if any(c in held for c in named):
                continue
            nearest = sorted(held, key=lambda v: (-similarity(named[0], v), v))[:NEAREST]
            found.append(StaleCodes(definition.term, column, objects, tuple(named), tuple(nearest),
                                    definition.definition))
    return found


def noted_stale(stale: Sequence[StaleCodes], definitions: Sequence[Definition], grants: GrantSet,
                columns: Sequence[Column]) -> list[StaleCodes]:
    """The findings to note beside `definitions` for this identity: each for a definition among them that still
    reads as it did when checked, where the identity is granted every object holding the column and reads each of
    those columns raw. The finding is about all of them, and a column of an object it cannot see, or one masked
    for it, is not its to be told about. A column not among `columns` cannot be judged here, so it is not noted."""
    shown = {(d.term, d.definition) for d in definitions}
    noted = []
    for found in stale:
        if (found.term, found.definition) not in shown or not all(grants.allows(o) for o in found.objects):
            continue
        held = [c for c in columns if c.object_id in found.objects and c.name.lower() == found.column.lower()]
        if held and all(read_raw(c, grants) for c in held):
            noted.append(found)
    return noted
