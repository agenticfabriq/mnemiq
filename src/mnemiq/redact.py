"""Credentials taken out of a source's own words before the model reads them (M96).

A source's error is shown to the model on purpose -- it is the repair's whole input -- and a
model can copy what it is shown into a string literal of its SQL, which reaches the caller. A
connection string has no repair value, so it is not shown: text the model never sees it cannot
launder. Two grammars, never a guess at what looks secret:

  * URI userinfo (RFC 3986): everything between `scheme://` and the last `@` before whitespace,
    `?` or `#` -- `postgresql://svc:pw@10.2.0.7/db` becomes `postgresql://***@10.2.0.7/db`. The
    last `@`, and across a `/`, so an unencoded `@` or `/` in a password is not left half-shown.
    A `@` in a URL's path is over-read as userinfo, which shows less, never more.
  * a password field: a key ending in `password`, `passwd` or `pwd` (so `PGPASSWORD` and
    `db_password` too, never `passwordless`), optionally quoted, then `=` or `:`, then its value
    read by the value's own grammar -- quoted ('...' or "..."), braced as ODBC writes `{a;b}` (with
    `}}` a literal brace), or a bare token running to the next UNESCAPED whitespace, as libpq reads
    an unquoted value -- so it may hold `=`, `;`, `&`, `,`, braces or quotes, and a backslash-escaped
    space does not end it. ODBC and a JDBC query end a value at `;` or `&`, libpq does not, and the text cannot say
    which it is: so a bare value is read to whitespace, and an ODBC or JDBC tail after the
    password (`;Server=s`) is scrubbed with it (Codex's review of #125: `password=abc;TOPSECRET`
    showed `TOPSECRET`). Whitespace after the separator is
    skipped, as libpq skips it -- whose grammar spells an empty value `''`, so in
    `password= host=h` the password IS `host=h`, and it is scrubbed. Where the grammar is
    ambiguous this reads more as secret, never less: a scrubber's error should hide repair input,
    not show a credential.

The server log keeps the full error (`ExecutionError.full_text`): it is the operator's record,
and the operator holds the credential anyway.

The access policy is the other thing a source's words can carry (Codex's review of #152). The source
runs the policy's rewrite, so its error can quote the predicate, and `withhold_policy_words` keeps
from the model whatever names a word only the policy wrote. Same log rule: the operator reads it all.
"""

from __future__ import annotations

import re

_URI_USERINFO = re.compile(r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.\-]*://)[^\s?#]+@")
_PASSWORD_KEY = re.compile(r"(?i)[\"']?\b\w*(?:password|passwd|pwd)[\"']?\s*[=:]")
_BARE_END = frozenset(" \t\r\n")


def scrub_credentials(text: str) -> str:
    """`text` with URI userinfo and password fields replaced by `***`."""
    text = _URI_USERINFO.sub(lambda m: f"{m['scheme']}***@", text)
    out: list[str] = []
    done = 0
    for key in _PASSWORD_KEY.finditer(text):
        if key.start() < done:
            continue
        start = key.end()
        value = start
        while value < len(text) and text[value].isspace():  # any, newlines too, as libpq
            value += 1
        end = _value_end(text, value)
        if end is None:
            continue
        out.append(text[done:value] + "***")
        done = end
    out.append(text[done:])
    return "".join(out)


def _value_end(text: str, value: int) -> int | None:
    """Where the value beginning at `value` ends, or None where the field has no value."""
    if value >= len(text):
        return None
    opener = text[value]
    if opener in "'\"":
        end = value + 1
        while end < len(text) and text[end] != opener:
            end += 2 if text[end] == "\\" else 1
        return min(end + 1, len(text))
    if opener == "{":
        end = value + 1
        while end < len(text):
            if text[end] == "}":
                if text[end + 1:end + 2] == "}":  # ODBC's escaped brace, inside the value
                    end += 2
                    continue
                return end + 1
            end += 1
        return len(text)
    end = value
    while end < len(text) and text[end] not in _BARE_END:
        end += 2 if text[end] == "\\" else 1  # libpq's backslash escape: `\ ` is in the value
    return min(end, len(text)) if end > value else None


# Postgres and DuckDB quote the statement after the message, as `LINE <n>: <window>` and a line
# holding only the caret under the error's position.
_ECHO = re.compile(r"^\s*LINE \d+:.*$|^\s*\^\s*$")


def withhold_policy_words(detail: str | None, *, asked: str | None, executed: str,
                          dialect: str) -> str | None:
    """`detail` as the model may read it when the source ran `executed` in place of `asked`.

    The source raised its error running the policy's rewrite, so its words can quote the policy:
    Postgres and DuckDB echo a window of the statement, MySQL quotes it after `near`, and DuckDB
    lists candidate names from the scope that failed. The policy's predicate names tables and
    literals the caller is not granted, and the prompt is not a place they may go. So the echo is
    cut, and the rest is withheld whole (None) if it names any identifier, function or literal that
    `executed` has and `asked` does not. Matched as whole words, case-insensitively: a remainder
    that names a policy-only word in any spelling loses the source's words, and the repair keeps
    our own sentence. Under-telling the model costs a retry; over-telling cannot be taken back.

    That includes GRANTED columns the caller never wrote, which the rewrite's derived table
    projects or its predicate reads. Deliberately: an error naming a column only the policy reads
    was raised in the policy's part, which runs over the rows the filter removes, so its values
    can be ones the caller may not see (`CAST(region AS INTEGER) > 0` as the policy quotes
    `'west'` from any row). The cost is DuckDB's candidate list for a mistyped column under a
    filter, and only on a stale snapshot: against a current one the decider refuses the name
    itself, in our own words, before the source sees it.

    With `asked == executed` the policy changed nothing and `detail` comes back untouched, so a
    benchmark, which clears every policy, sends the same bytes. With no record of `asked`, or an
    `executed` that will not parse, the policy's words cannot be told from the caller's, and
    nothing of the source's is kept.
    """
    if detail is None or asked == executed:
        return detail
    if asked is None:
        return None
    try:
        policy_words = _words(executed, dialect) - _words(asked, dialect)
    except Exception:  # sqlglot raises several types; any of them means "cannot tell"
        return None
    kept = "\n".join(line for line in detail.splitlines() if not _ECHO.match(line)).strip()
    if not kept or any(_mentions(kept, word) for word in policy_words):
        return None
    return kept


def _words(sql: str, dialect: str) -> set[str]:
    """Every identifier, function name and literal value `sql` spells, folded to lower case."""
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read=dialect)
    words = {node.name for node in tree.find_all(exp.Identifier)}
    words |= {node.name for node in tree.find_all(exp.Literal)}
    words |= {node.name for node in tree.find_all(exp.Anonymous)}
    return {w.lower() for w in words if w}


def _mentions(text: str, word: str) -> bool:
    # A word boundary only where the word itself starts or ends with a word character: a literal
    # like `a-b` or `%x%` is still found inside other punctuation.
    left = r"(?<!\w)" if re.match(r"\w", word) else ""
    right = r"(?!\w)" if re.search(r"\w$", word) else ""
    return re.search(f"(?i){left}{re.escape(word)}{right}", text) is not None
