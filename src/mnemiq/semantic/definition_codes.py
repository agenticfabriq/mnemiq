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
  `NOT IN ('X')`, "is not 'X'", "other than", "except", "excluding", "exclude ... = 'X'", "'X' ... are excluded") is
  filtered out by a query that follows the definition, which keeps every row whether or not the data holds it, so a
  definition that only excludes is never reported. A nearby negation that still selects ("not only 'X' but also",
  "do not omit ... 'X'", "do not exclude ... 'X'") is not an exclusion, and "exclude" reaches only its own predicate.
- A code counts for a column only when it has the shape of one of the column's values: capitals, small letters and
  digits in the same places. `'from May to June'` is never a code for a column of `AB12`s.
- A column is judged only when every column of that name in the snapshot is indexed: a code may be in the one whose
  values were not read, which is unknown, not absent.

The value index already leaves out keys, personal data and columns with too many values, so none of those is judged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from mnemiq.contract import Snapshot
from mnemiq.semantic.textmatch import similarity

# Opened after a non-word character, closed before one, never starting or ending in whitespace: "'AX' or 'AY'" yields AX
# and AY, not " or ", and "the lot's mode='-A' or '-G'" yields -A and -G, not "s mode=".
_QUOTED = re.compile(r"(?<!\w)'([^'\s](?:[^'\n]{0,62}[^'\s])?)'(?!\w)|(?<!\w)\"([^\"\s](?:[^\"\n]{0,62}[^\"\s])?)\"(?!\w)")
# A clause ends at . ; : ? a newline, or a ! that does not begin !=.
_CLAUSE_END = re.compile(r"[.;:?\n]|!(?!=)")
# An explicit exclusion just before the code, in its clause...
_EXCLUDED_BEFORE = re.compile(
    r"(?:<>|!=|\bnot\s+in\s*\((?:[^()]*,)?|\b(?:is|are|was|were)\s+(?:not|never)|\bnot\s+equals?(?:\s+to)?|\bother\s+than"
    r"|\bexcept(?:\s+for)?|\bexcluding|\bbut\s+not"
    # "exclude ... =", not negated, and reaching back over no other quoted code: its own predicate only.
    r"|(?<!\bnot )(?<!\bnever )(?<!n't )\bexclud\w*\b[^.;:?\n'\"]*?=)\s*$", re.IGNORECASE)
# ...or just after it. Both read a coordinated list ("'X', 'Y' and 'Z'") as one mention.
_EXCLUDED_AFTER = re.compile(
    r"^\s*\)?\s*(?:rows?\s+)?(?:are|is)\s+(?:excluded|ignored|dropped|removed|left\s+out|filtered\s+out)\b", re.IGNORECASE)
# What may stand between two codes of one list: a comma, "and", "or", or both.
_LIST_GAP = re.compile(r"\s*(?:,\s*)?(?:(?:and|or)\s+)?", re.IGNORECASE)
_BARE = re.compile(r"(?<![\w'\"-])[A-Z0-9][A-Z0-9_-]*(?![\w'\"-])")
NEAREST = 8  # real values quoted, as many as the value check quotes


@dataclass(frozen=True)
class StaleCodes:
    """A definition that names `codes` for `column`, none of which any column of that name holds."""

    term: str
    column: str
    objects: tuple[str, ...]  # the objects whose column was read
    codes: tuple[str, ...]
    nearest: tuple[str, ...]  # the column's real values most like the first code, best first


def _shape(text: str) -> str:
    return "".join("A" if c.isupper() else "a" if c.islower() else "9" if c.isdigit() else c for c in text)


def _identifier_like(name: str) -> bool:
    return "_" in name or any(c.isdigit() for c in name)


def _names(column: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(column)}(?!\w)", text, re.IGNORECASE) is not None


def _excluded(text: str, start: int, end: int) -> bool:
    """Whether the mention at `text[start:end]` sits in an explicit exclusion within its own clause."""
    before = _CLAUSE_END.split(text[:start])[-1]
    after = _CLAUSE_END.split(text[end:], maxsplit=1)[0]
    return _EXCLUDED_BEFORE.search(before) is not None or _EXCLUDED_AFTER.search(after) is not None


def _codes(text: str, names: set[str]) -> list[str]:
    """The codes the text selects on: each quoted or bare code with at least one mention outside an exclusion.

    Consecutive mentions joined only by a comma, "and" or "or" are one list, read as one mention: "'X' and 'Y' are
    excluded" excludes both, and "exclude ... = 'X' or 'Y'" both."""
    quoted = [(m.group(1) or m.group(2), m.start(), m.end()) for m in _QUOTED.finditer(text)]
    bare = [(m.group(0), m.start(), m.end()) for m in _BARE.finditer(text)
            if m.group(0).lower() not in names and any(c.isalpha() for c in m.group(0))
            and any(c.isdigit() for c in m.group(0))]
    lists: list[list[tuple[str, int, int]]] = []
    for mention in sorted(quoted + bare, key=lambda m: m[1]):
        if lists and _LIST_GAP.fullmatch(text[lists[-1][-1][2]:mention[1]]):
            lists[-1].append(mention)
        else:
            lists.append([mention])
    return [code for group in lists if not _excluded(text, group[0][1], group[-1][2]) for code, _, _ in group]


def stale_codes(snapshot: Snapshot, values) -> list[StaleCodes]:
    """Each definition and indexed column where the definition names codes the column holds none of.

    `values` is the value index (`ValueIndex`), read through `has` and `values`; an index that holds nothing
    (`NoValueIndex`) judges nothing."""
    read = getattr(values, "values", None)
    if read is None:
        return []
    by_name: dict[str, list[tuple[str, str]]] = {}
    for col in snapshot.columns:
        by_name.setdefault(col.name.lower(), []).append((col.object_id, col.name))
    names = set(by_name) | {col.object_id.lower() for col in snapshot.columns}
    judged: dict[str, tuple[str, tuple[str, ...], list[str]]] = {}
    for key, cols in by_name.items():
        if not _identifier_like(key) or not all(values.has(obj, name) for obj, name in cols):
            continue
        held = sorted({v for obj, name in cols for v in read(obj, name)})
        if held:
            judged[key] = (cols[0][1], tuple(sorted(obj for obj, _ in cols)), held)
    found: list[StaleCodes] = []
    for definition in snapshot.definitions:
        text = definition.definition
        codes = _codes(text, names)
        for key, (column, objects, held) in sorted(judged.items()):
            if not _names(column, text):
                continue
            shapes = {_shape(v) for v in held}
            named = sorted({c for c in codes if _shape(c) in shapes})
            if not named or any(c in held for c in named):
                continue
            nearest = sorted(held, key=lambda v: (-similarity(named[0], v), v))[:NEAREST]
            found.append(StaleCodes(definition.term, column, objects, tuple(named), tuple(nearest)))
    return found
