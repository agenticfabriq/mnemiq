"""Credentials taken out of a source's own words before the model reads them (M96).

A source's error is shown to the model on purpose -- it is the repair's whole input -- and a
model can copy what it is shown into a string literal of its SQL, which reaches the caller. A
connection string has no repair value, so it is not shown: text the model never sees it cannot
launder. Two grammars, never a guess at what looks secret:

  * URI userinfo (RFC 3986): everything between `scheme://` and the last `@` before the host --
    `postgresql://svc:pw@10.2.0.7/db` becomes `postgresql://***@10.2.0.7/db`. The last `@`, so an
    unencoded `@` inside a password is not left half-shown.
  * a keyword=value password, as libpq, ODBC and a JDBC query string spell it -- `password=`,
    `passwd=`, `pwd=`, any case, the value quoted or bare.

The server log keeps the full error: it is the operator's record, and the operator holds the
credential anyway.
"""

from __future__ import annotations

import re

_URI_USERINFO = re.compile(r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.\-]*://)[^\s/?#]+@")
_KEYWORD_SECRET = re.compile(
    r"(?i)\b(?P<key>password|passwd|pwd)(?P<eq>\s*=\s*)"
    r"(?P<value>'(?:[^'\\]|\\.)*'|\"[^\"]*\"|[^\s;&'\"]+)"
)


def scrub_credentials(text: str) -> str:
    """`text` with URI userinfo and keyword=value passwords replaced by `***`."""
    text = _URI_USERINFO.sub(lambda m: f"{m['scheme']}***@", text)
    return _KEYWORD_SECRET.sub(lambda m: f"{m['key']}{m['eq']}***", text)
