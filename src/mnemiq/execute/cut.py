"""An answer cut to fit one response of its source, marked on its own table.

A source that sends a statement's answer in one response of a bounded size (Superset's MCP service does) can send only
the answer's first rows. The table those rows make carries how many there were in its schema's metadata, so the mark
travels wherever the table does: through the result cache, which keeps a table's metadata, through a vote between
candidates, which compares tables without it, and into the answer, which says the rows stop there.
"""

from __future__ import annotations

import pyarrow as pa

_SENT = b"mnemiq.sent"


def mark_sent(table: pa.Table, sent: int) -> pa.Table:
    """`table`, marked as the first `sent` rows of an answer its source cut to fit one response."""
    return table.replace_schema_metadata({**(table.schema.metadata or {}), _SENT: str(int(sent)).encode()})


def sent_of(table: pa.Table) -> int | None:
    """How many rows the source sent of an answer it cut to fit one response, or None for a table it sent whole."""
    raw = (table.schema.metadata or {}).get(_SENT)
    return int(raw) if raw is not None else None
