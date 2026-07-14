from __future__ import annotations

import pyarrow as pa


def _cell(value: object, max_cell: int) -> str:
    if value is None:
        return "NULL"  # not "None": the model is reading SQL results, not Python
    text = str(value)
    return text if len(text) <= max_cell else text[: max_cell - 1] + "…"


def render_result(table: pa.Table, max_rows: int = 50, max_cell: int = 120) -> str:
    """Render rows for a prompt: bounded, and loud about what it left out.

    Silently showing 50 of 1000 rows invites the model to summarize a partial view as if it
    were the whole -- a confidently wrong answer, which is the one failure mode that matters.
    """
    columns = table.schema.names
    total = table.num_rows
    if total == 0:
        return f"columns: {', '.join(columns)}\n(0 rows)"

    shown = min(total, max_rows)
    rows = table.slice(0, shown).to_pylist()

    lines = [" | ".join(columns)]
    lines += [" | ".join(_cell(row[c], max_cell) for c in columns) for row in rows]

    if shown < total:
        lines.append(f"... showing {shown} of {total} rows (truncated)")
    else:
        lines.append(f"({total} rows)")
    return "\n".join(lines)
