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

The access policy is the other thing a source's words can carry (Codex's reviews of #152 and #153).
The source runs the policy's rewrite, so under a policy `policy_safe_detail` lets the model read the
engine's error class and nothing else of the source's. Same log rule: the operator reads it all.
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


# DuckDB begins every message with its exception type, letters and spaces then ` Error:`; Oracle's
# driver begins with the server's or its own code. Read only at the very start, where the engine
# writes them, so neither pattern can reach into the message that follows.
_ERROR_CLASS = (re.compile(r"(?:[A-Za-z]+ ){0,3}[A-Za-z]+ Error(?=:)"), re.compile(r"(?:ORA|DPY)-\d{4,5}\b"))


def policy_safe_detail(detail: str | None, *, asked: str | None, executed: str) -> str | None:
    """`detail` as the model may read it when the source ran `executed` in place of `asked`.

    Under a policy, the engine's error class (`Conversion Error`, `ORA-01722`) and nothing more;
    None where the text begins with no class this recognises. Fail-closed, because no reading of
    the rest can tell data from words. #153 tried: it cut the statement's echo and withheld text
    naming any word only the policy wrote, and Codex's review of it measured the hole -- a policy
    that raises on data with its column wrapped (`CAST(UPPER(region) AS INTEGER) > 0`) gets
    "Could not convert string 'SECRET_HIDDEN_VALUE' to INT32", a row the caller may not see, naming
    no column at all. DuckDB prints values unquoted too ("with value 31337"), and Oracle runs the
    caller's own predicate on rows the policy removes and quotes a character of them (M160), so
    neither the policy's words nor quotation marks find the data. A class is the engine's fixed
    vocabulary: it says what kind of rewrite to try and carries no row.

    Costs the repair the source's detail on every statement a policy changed -- the candidate list,
    the column a cast failed on -- and only there: most of what a repair fixes is refused by the
    decider in our own words before the source sees it. The full text is in the operator's log on
    both paths. It does not close M160's channel, which is whether an error fires at all; the
    policy fence that bounds the caller's predicates is the rewrite's, and a policy that raises on
    data stays M159's to refuse.

    `asked == executed`: the policy changed nothing and `detail` comes back untouched, so a
    benchmark, which clears every policy, sends the same bytes. No record of `asked` cannot say so,
    and counts as changed.
    """
    if detail is None or (asked is not None and asked == executed):
        return detail
    for pattern in _ERROR_CLASS:
        found = pattern.match(detail)
        if found:
            return f"Error class: {found.group(0)}. The rest of the source's message is not shown."
    return None
