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
    `}}` a literal brace), or a bare token running to whitespace, `;`, `&` or a quote, so it may
    hold `=`, `,` or braces (a base64 password ends in `=`). Whitespace after the separator is
    skipped, as libpq skips it -- whose grammar spells an empty value `''`, so in
    `password= host=h` the password IS `host=h`, and it is scrubbed. Where the grammar is
    ambiguous this reads more as secret, never less: a scrubber's error should hide repair input,
    not show a credential.

The server log keeps the full error (`ExecutionError.full_text`): it is the operator's record,
and the operator holds the credential anyway.
"""

from __future__ import annotations

import re

_URI_USERINFO = re.compile(r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.\-]*://)[^\s?#]+@")
_PASSWORD_KEY = re.compile(r"(?i)[\"']?\b\w*(?:password|passwd|pwd)[\"']?\s*[=:]")
_BARE_END = frozenset(" \t\r\n;&'\"")


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
        end += 1
    return end if end > value else None
