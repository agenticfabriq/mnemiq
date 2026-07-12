from __future__ import annotations

import re


def translate(tsql: str) -> list[str]:
    """Translate the ACME T-SQL DDL into executable Postgres CREATE TABLE statements.

    - `datetime` -> `timestamp`
    - drop `ASC` in key definitions
    - drop PRIMARY KEY and FOREIGN KEY constraints (keys/relationships are recovered
      by enrichment; the sample CSVs have null/duplicate key values, and constraints
      impose load ordering — the fixture only needs the tables + data, queryable)
    - drop `NOT NULL` (the sample CSVs don't populate every non-null column; the
      fixture is for querying, not integrity enforcement)
    - terminate each statement with `;`
    """
    blocks = re.split(r"(?=CREATE\s+TABLE\b)", tsql, flags=re.IGNORECASE)
    stmts: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b.upper().startswith("CREATE TABLE"):
            continue
        b = re.sub(r"\bdatetime\b", "timestamp", b, flags=re.IGNORECASE)
        b = re.sub(r"\s+ASC\b", "", b, flags=re.IGNORECASE)
        b = re.sub(r"\bNOT\s+NULL\b", "", b, flags=re.IGNORECASE)
        b = "\n".join(
            line
            for line in b.splitlines()
            if not re.match(r"\s*,?\s*(PRIMARY|FOREIGN)\s+KEY", line, flags=re.IGNORECASE)
        )
        # a dropped trailing FK can leave a comma dangling before the closing paren
        b = re.sub(r",(\s*)\)", r"\1)", b)
        b = b.rstrip().rstrip(";") + ";"
        stmts.append(b)
    return stmts
