from __future__ import annotations

from itertools import combinations

import pyarrow as pa

from mnemiq.execute.resultset import (
    _match_rows,
    _rows,
    facts_cells_match,
    grade_cells_match,
    normalize,
)

# normalize is re-exported: harness.py imports it from here. _rows/_match_rows back
# results_match below; the shared definitions live in the engine's resultset module.
__all__ = ["normalize", "results_match"]

# The definitions these two readings implement are the grading contract agreed with
# beacon on 2026-08-07: beacon `docs/grading.md`. That document is the agreement; this
# file is one of its two implementations. Change the meaning there, not here -- two
# implementations restating a definition is how they come to disagree.


def _distinct_rows(rows: list[list[object]]) -> list[list[object]]:
    """Collapse duplicate rows, keeping first-occurrence order.

    Mirrors beacon's `_distinct_rows` deliberately, `repr` key and all: this is the same rule
    graded in two repos, and a second implementation that merely looks equivalent is how the
    two come to disagree on a row neither author thought about.
    """
    seen: set[str] = set()
    out: list[list[object]] = []
    for row in rows:
        key = repr(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def results_match(
    gold: pa.Table,
    candidate: pa.Table,
    allow_extra_columns: bool = True,
    duplicate_rows_insignificant: bool = False,
) -> bool:
    """Do these two result sets state the same facts?

    Not: are they the same query, the same column names, the same row order, or the same
    formatting. A different query that returns the right answer is correct -- that is the
    whole point of result-based grading (spec 6.9). What must match is the data.

    With allow_extra_columns=False the candidate must have exactly the gold's columns; by
    default it may carry extra context columns (ACME's conversational questions -- the date
    next to the policy number it was asked for).

    **The strict reading is not BIRD execution accuracy, and this docstring used to say it
    was.** BIRD's `calculate_ex` is one line, `set(pred) == set(gold)` over raw driver tuples,
    and this differs from it in three places that do not share a sign:

      * STRICTER on row multiplicity -- `_match_rows` is a multiset comparison and refuses on
        a row-count difference, where a set collapses duplicates. That one was a VIOLATION of
        the grading contract rather than a choice, and `duplicate_rows_insignificant` is the
        fix: beacon's `docs/grading.md` has BIRD items declare it "so that exact on BIRD is
        the number the leaderboard publishes", measured there at 24 cases in 2,899 -- 0.83
        points understated. It is per item, so a benchmark that does not declare it keeps the
        multiset reading, and it collapses in BOTH metrics, as beacon does.
      * MORE TOLERANT on float cells -- `grade_cells_match` carries GRADE_ABS_TOL/GRADE_REL_TOL
        where a set comparison carries none. Recorded and deliberate: beacon's runners cross
        engines, where the same quantity arrives with different float noise.
      * MORE TOLERANT on normalisation -- `normalize` strips and coerces both sides, where
        BIRD normalises nothing.

    So a number graded here is comparable to a published one only when the flag is set, and
    even then it carries the two tolerance divergences. Report which rule produced a figure;
    the difference has no sign, so no number can be adjusted for it. Register M105.

    This flag is really which of the two metrics is being asked for, so it also selects
    the cell reading. Got-facts means "the information gold wants, with extra columns or
    different formatting", and a value printed to fewer decimals is formatting, so the
    tolerant reading accepts a rounding either way (facts_cells_match) while the strict
    one does not. The rule is beacon's, ported rather than reinvented: one definition
    graded two ways is how the same column comes to mean different things per benchmark.

    The numeric bounds are beacon's too (grade_cells_match), replacing a 1e-2 band that
    was doing the rounding rule's job badly: 1% called 1038.15 and 1039.32 the same
    answer while still needing luck on a genuine rounding.
    """
    cell_match = facts_cells_match if allow_extra_columns else grade_cells_match
    if allow_extra_columns:
        # One-directional tolerance: the candidate may ADD context columns but never omit a
        # gold column. Extra columns cannot rescue wrong rows -- every gold row still matches.
        if gold.num_columns > candidate.num_columns:
            return False
        column_choices = combinations(range(candidate.num_columns), gold.num_columns)
    else:
        if gold.num_columns != candidate.num_columns:
            return False
        column_choices = [tuple(range(candidate.num_columns))]

    gold_rows = _rows(gold)
    if duplicate_rows_insignificant:
        gold_rows = _distinct_rows(gold_rows)
    for keep in column_choices:
        projected = candidate.select(list(keep))
        candidate_rows = _rows(projected)
        if duplicate_rows_insignificant:
            candidate_rows = _distinct_rows(candidate_rows)
        if _match_rows(gold_rows, candidate_rows, 0.0, cell_match):
            return True

        # Column order is presentation, so got-facts retries with each row's cells sorted
        # into a canonical order -- `SELECT k, count(*)` and `SELECT count(*), k` carry
        # the same information. Exact match does NOT: BIRD's evaluator compares row
        # tuples position-wise, so the strict number has to stay the one the leaderboard
        # publishes. This retry running on both readings is what made mnemiq's own BIRD
        # "correct" slightly generous against the published metric.
        if not allow_extra_columns:
            continue

        def sort_cells(rows: list[list[object]]) -> list[list[object]]:
            return [sorted(row, key=repr) for row in rows]

        # Re-collapse AFTER sorting: two rows distinct by column order become identical
        # once each row's cells are canonicalised, so a dedupe done only before the loop
        # leaves duplicates this branch created. Measured on the real module -- gold
        # `{a:[1,2], b:[2,1]}` against candidate `{a:[1], b:[2]}` answered False while the
        # same fact spelled `{a:[1,1], b:[2,2]}` answered True, which is the docstring's
        # "collapses in BOTH metrics" failing on exactly one branch.
        sorted_gold, sorted_candidate = sort_cells(gold_rows), sort_cells(candidate_rows)
        if duplicate_rows_insignificant:
            sorted_gold = _distinct_rows(sorted_gold)
            sorted_candidate = _distinct_rows(sorted_candidate)
        if _match_rows(sorted_gold, sorted_candidate, 0.0, cell_match):
            return True
    return False
