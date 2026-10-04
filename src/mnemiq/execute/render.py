from __future__ import annotations

import pyarrow as pa

# Long enough that ordinary values -- names, codes, dates, amounts -- are never cut, so the
# marker means something when it does appear. 120 silently halved schema and description
# columns, and a model cannot tell a value that ends from one that was cut.
MAX_CELL = 240
# Columns are bounded like rows and cells (M130). A wide `SELECT *` -- 965 columns on a design
# partner's shape -- rendered every column into the judge's, the selector's and the answer-writer's
# prompts, and could outgrow the window the generation prompt had just been fitted to. All three
# share this bound on purpose: the answer writer describes the columns the judge read, and the user
# still sees every column, because the answer's display preview (`result_preview`) is not bounded
# here. Gold results never come near it -- at most 6 columns across BIRD dev (1,534 cases) and
# mini-dev (500), 19 on KaggleDBQA -- so only a candidate that selects a whole wide table is cut
# (BIRD's widest: 115). It removes the wide-table blowup; it does not guarantee a fit, since rows
# times cells still multiply.
MAX_COLS = 50


def _cell(value: object, max_cell: int) -> tuple[str, bool]:
    """The rendered cell, and whether it was shortened."""
    if value is None:
        return "NULL", False  # not "None": the model is reading SQL results, not Python
    text = str(value)
    if len(text) <= max_cell:
        return text, False
    return text[: max_cell - 1] + "…", True


def render_result(table: pa.Table, max_rows: int = 50, max_cell: int = MAX_CELL,
                  cut_at: int | None = None, max_cols: int | None = MAX_COLS) -> str:
    """Render rows for a prompt: bounded, and loud about what it left out.

    Silently showing 50 of 1000 rows invites the model to summarize a partial view as if it
    were the whole -- a confidently wrong answer, which is the one failure mode that matters.

    The same argument applies within a row, and used not to be made: cells were cut to a
    bare `…` with nothing to say so. A reader could not tell a truncated value from the
    data, and neither could the model -- it read the ellipses as evidence and reported rows
    as truncated when only cells were. Both bounds are declared now.

    Columns too, past `max_cols`: the first ones are shown and the rest counted, never listed --
    listing 900 hidden names would undo the bound.
    """
    every = table.schema.names
    columns = every[:max_cols] if max_cols is not None and len(every) > max_cols else every
    hidden = len(every) - len(columns)
    hidden_note = (f"... showing the first {len(columns)} of {len(every)} columns; the other "
                   f"{hidden} are not shown -- do not describe, compare or count them")
    total = table.num_rows
    if total == 0:
        return f"columns: {', '.join(columns)}\n(0 rows)" + (f"\n{hidden_note}" if hidden else "")

    shown = min(total, max_rows)
    rows = table.slice(0, shown).to_pylist()

    lines = [" | ".join(columns)]
    shortened = False
    for row in rows:
        rendered = [_cell(row[c], max_cell) for c in columns]
        shortened = shortened or any(cut for _, cut in rendered)
        lines.append(" | ".join(text for text, _ in rendered))

    if cut_at is not None:
        # Not a total: the query stopped at the engine's limit and may have more (M118).
        lines.append(f"... showing {shown} rows; the query stopped at the {cut_at:,}-row limit, "
                     "so there may be more -- do not report a total or a count of these rows")
    elif shown < total:
        lines.append(f"... showing {shown} of {total} rows (truncated)")
    else:
        lines.append(f"({total} rows)")
    if hidden:
        lines.append(hidden_note)
    if shortened:
        lines.append(
            f"... some values were longer than {max_cell} characters and end in `…`; "
            "those are shortened for this prompt, not the stored value"
        )
    return "\n".join(lines)
