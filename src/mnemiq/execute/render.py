from __future__ import annotations

import pyarrow as pa

# Long enough that ordinary values -- names, codes, dates, amounts -- are never cut, so the
# marker means something when it does appear. 120 silently halved schema and description
# columns, and a model cannot tell a value that ends from one that was cut.
MAX_CELL = 240


def _cell(value: object, max_cell: int) -> tuple[str, bool]:
    """The rendered cell, and whether it was shortened."""
    if value is None:
        return "NULL", False  # not "None": the model is reading SQL results, not Python
    text = str(value)
    if len(text) <= max_cell:
        return text, False
    return text[: max_cell - 1] + "…", True


def render_result(table: pa.Table, max_rows: int = 50, max_cell: int = MAX_CELL) -> str:
    """Render rows for a prompt: bounded, and loud about what it left out.

    Silently showing 50 of 1000 rows invites the model to summarize a partial view as if it
    were the whole -- a confidently wrong answer, which is the one failure mode that matters.

    The same argument applies within a row, and used not to be made: cells were cut to a
    bare `…` with nothing to say so. A reader could not tell a truncated value from the
    data, and neither could the model -- it read the ellipses as evidence and reported rows
    as truncated when only cells were. Both bounds are declared now.
    """
    columns = table.schema.names
    total = table.num_rows
    if total == 0:
        return f"columns: {', '.join(columns)}\n(0 rows)"

    shown = min(total, max_rows)
    rows = table.slice(0, shown).to_pylist()

    lines = [" | ".join(columns)]
    shortened = False
    for row in rows:
        rendered = [_cell(row[c], max_cell) for c in columns]
        shortened = shortened or any(cut for _, cut in rendered)
        lines.append(" | ".join(text for text, _ in rendered))

    if shown < total:
        lines.append(f"... showing {shown} of {total} rows (truncated)")
    else:
        lines.append(f"({total} rows)")
    if shortened:
        lines.append(
            f"... some values were longer than {max_cell} characters and end in `…`; "
            "those are shortened for this prompt, not the stored value"
        )
    return "\n".join(lines)
