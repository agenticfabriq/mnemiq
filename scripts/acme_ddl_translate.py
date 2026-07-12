from __future__ import annotations

import re


def translate(tsql: str) -> list[str]:
    # Split into CREATE TABLE blocks (statements are separated by blank lines, no ';').
    blocks = re.split(r"(?=CREATE\s+TABLE\b)", tsql, flags=re.IGNORECASE)
    stmts: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b.upper().startswith("CREATE TABLE"):
            continue
        b = re.sub(r"\bdatetime\b", "timestamp", b, flags=re.IGNORECASE)
        b = re.sub(r"\s+ASC\b", "", b, flags=re.IGNORECASE)
        b = b.rstrip().rstrip(";") + ";"
        stmts.append(b)
    return stmts
