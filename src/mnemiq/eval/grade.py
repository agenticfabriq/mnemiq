from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
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
__all__ = ["GotFactsUndecided", "normalize", "results_match"]

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


# How many surviving column choices got-facts will try before it says it cannot decide. On
# the design-partner answers at most 13 survived; past this the answer is pathological (dozens
# of flag-like columns on a wrong answer), and a visible "undecided" beats a hang or a silent
# WRONG -- the bound that scored 19 right answers wrong was exactly such a silent WRONG.
MAX_CHOICES = 10_000


# How many cell comparisons the row checks may spend in one grading. A check is the engine's
# greedy `_match_rows`, quadratic in rows: two copies of 2,100 rows in reverse order cost 2.2
# million comparisons in ONE check. Answers are capped at 1,000 rows (500,000 at worst per check),
# so this allows dozens of full checks and stops a pathological grading instead of stalling it.
MAX_CHECK_STEPS = 20_000_000

# How many value lookups and cell comparisons the pruning itself may spend. It reads each DISTINCT value once and
# stops at the first match, so ordinary answers spend a few thousand; this bounds the
# pathological ones before the choice limit is ever reached.
MAX_PRUNING_STEPS = 2_000_000


class GotFactsUndecided(Exception):
    """Got-facts could not decide: more column choices survived pruning than it will try, the
    pruning or the row checks ran out of budget, or the only choices left meet a cell pair the
    readings cannot compare. Not a verdict: a caller grading a case records it as an error, and a caller
    re-checking a stored label keeps the label and marks it not re-checked -- none turns it
    into a new WRONG."""


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
    elif gold.num_columns != candidate.num_columns:
        return False

    gold_rows = _rows(gold)
    if duplicate_rows_insignificant:
        gold_rows = _distinct_rows(gold_rows)
    # Normalized ONCE and projected by index: normalize() reads one cell and nothing else, so
    # projecting after normalizing is exact, and the projections below never re-read the table.
    all_rows = _rows(candidate)

    if not allow_extra_columns:
        candidate_rows = _distinct_rows(all_rows) if duplicate_rows_insignificant else all_rows
        return _match_rows(gold_rows, candidate_rows, 0.0, cell_match)

    # Without the declaration both readings compare multisets, and projecting keeps the row
    # count, so a different count fails every projection: say so once instead of trying them all.
    if not duplicate_rows_insignificant and len(gold_rows) != len(all_rows):
        return False

    # Column order is presentation, so got-facts also compares each row with its cells sorted
    # into a canonical order -- `SELECT k, count(*)` and `SELECT count(*), k` carry the same
    # information. Exact match does NOT: BIRD's evaluator compares row tuples position-wise.
    sorted_gold = [sorted(row, key=repr) for row in gold_rows]
    if duplicate_rows_insignificant:
        # Re-collapse AFTER sorting: two rows distinct by column order become identical once
        # each row's cells are canonicalised (gold `{a:[1,2], b:[2,1]}` against candidate
        # `{a:[1], b:[2]}` is one fact either way).
        sorted_gold = _distinct_rows(sorted_gold)
    all_keys = [[repr(cell) for cell in row] for row in all_rows]

    work = _Work(MAX_PRUNING_STEPS, candidate.num_columns, gold.num_columns)
    check_work = _Work(MAX_CHECK_STEPS, candidate.num_columns, gold.num_columns,
                       what="row checks", unit="cell comparisons")

    def counted_match(a: object, b: object) -> bool:
        check_work.spend()
        return cell_match(a, b)

    tried = 0
    uncomparable = False

    def spend() -> None:
        nonlocal tried
        tried += 1
        if tried > MAX_CHOICES:
            raise GotFactsUndecided(
                f"more than {MAX_CHOICES:,} column choices survive pruning "
                f"({candidate.num_columns} candidate columns, {gold.num_columns} gold)")

    for keep in _positional_choices(gold_rows, all_rows, gold.num_columns,
                                    candidate.num_columns, cell_match, work):
        spend()
        candidate_rows = [[row[i] for i in keep] for row in all_rows]
        if duplicate_rows_insignificant:
            candidate_rows = _distinct_rows(candidate_rows)
        verdict = _checks(gold_rows, candidate_rows, counted_match)
        if verdict:
            return True
        uncomparable |= verdict is None
    for keep in _sorted_choices(gold_rows, all_rows, gold.num_columns, candidate.num_columns,
                                cell_match, work):
        spend()
        sorted_candidate = [
            [value for _, value in sorted(((keys[i], row[i]) for i in keep), key=lambda p: p[0])]
            for row, keys in zip(all_rows, all_keys, strict=True)
        ]
        if duplicate_rows_insignificant:
            sorted_candidate = _distinct_rows(sorted_candidate)
        verdict = _checks(sorted_gold, sorted_candidate, counted_match)
        if verdict:
            return True
        uncomparable |= verdict is None
    if uncomparable:
        # Some choice met a pair the readings cannot compare and none matched without one: that
        # is not a mismatch, so it is not a verdict either.
        raise GotFactsUndecided("a cell pair the readings cannot compare (an infinity, or a "
                                "magnitude too large to round) decides the remaining choices")
    return False


def _checks(gold_rows: list[list[object]], candidate_rows: list[list[object]],
            cell_match: CellMatch) -> bool | None:
    """`_match_rows` on one choice, or None when it meets a cell pair the readings cannot
    compare (an infinity, or a magnitude Decimal cannot quantize, meeting the rounding rule).
    The full search raised on such a pair if it got that far; going on instead keeps every
    verdict it reached (it reached them before any such pair), finds a match it crashed short
    of, and leaves undecided -- never WRONG -- what only such pairs could decide."""
    try:
        return _match_rows(gold_rows, candidate_rows, 0.0, cell_match)
    except ArithmeticError:
        return None


# -- which column choices can possibly match -----------------------------------------------------
#
# Got-facts asks whether SOME choice of the candidate's columns, one per gold column, states
# gold's rows. Trying every choice is C(candidate columns, gold columns): 4e12 for a 74-column
# answer to an 11-column gold, and the search hung (register M113). Each pruning rule below is a
# condition every matching choice must meet, so dropping a choice that fails it cannot change
# the verdict -- the survivors are still checked by the same `_match_rows` as before.


CellMatch = Callable[[object, object], bool]


class _Work:
    """A budget of steps for one grading -- the pruning's, or the row checks' -- that raises
    GotFactsUndecided when spent, so no part of got-facts can stall a run."""

    def __init__(self, limit: int, candidate_cols: int, gold_cols: int,
                 what: str = "pruning", unit: str = "lookups and cell comparisons") -> None:
        self.limit = self.left = limit
        self.what, self.unit = what, unit
        self.shape = f"{candidate_cols} candidate columns, {gold_cols} gold"

    def spend(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise GotFactsUndecided(
                f"{self.what} spent more than {self.limit:,} {self.unit} ({self.shape})")


class _ValueIndex:
    """Answers "does any value here match `x`?" without comparing `x` to every value.

    Exact for both grading cell readings, which accept a pair only when the values are equal,
    when an infinity meets any number (the tolerance reads `inf <= 1e-5 * inf` as true), or when
    both are ordinary floats and either lie within the numeric tolerance (at most 5e-7, or 1e-5 of the
    gold value) or one is the other rounded to 0-6 decimal places (so at most half a unit of the
    last place kept, plus 1e-9). Those are the only places a match can hide, so the index looks
    there and lets `cell_match` decide each pair it finds. Values are held once each, and a pair
    `cell_match` cannot compare counts as a possible match: keeping a column only costs time.
    """

    def __init__(self, values: list[object], work: _Work) -> None:
        self._work = work
        # NaN is left out: a set finds it by identity, but the readings never call NaN equal.
        self._exact = {v for v in values
                       if (v is None or _hashable(v)) and not (isinstance(v, float) and math.isnan(v))}
        floats = [v for v in values if isinstance(v, float)]
        self._unsure = [v for v in floats if _unsure(v)]
        self._floats = sorted({v for v in floats if not _unsure(v)})
        self._fractional = [v for v in self._floats if not v.is_integer()]

    def any_match(self, x: object, cell_match: CellMatch, x_is_gold: bool) -> bool:
        def ok(v: object) -> bool:
            self._work.spend()
            try:
                return cell_match(x, v) if x_is_gold else cell_match(v, x)
            except ArithmeticError:
                return True  # the full search would raise on this pair: keep it, let the check decide

        self._work.spend()  # every lookup counts, not only the comparisons it leads to
        nan = isinstance(x, float) and math.isnan(x)
        # Equal values always match under both readings (None to None, 1.0 to True included).
        if not nan and (x is None or _hashable(x)) and x in self._exact:
            return True
        if not isinstance(x, float):
            return False  # anything else matches only by equality
        # The values the readings meet unpredictably are compared outright, every time.
        if any(ok(v) for v in self._unsure):
            return True
        if _unsure(x):
            # NaN against an ordinary number neither matches nor raises; an infinity or a huge
            # value may meet any of them (the tolerance, or a raise), so look at every one.
            return False if nan else any(ok(v) for v in self._floats)
        # Numeric tolerance: 1e-5 of the GOLD value, so widen a little when x is the candidate.
        # Two whole numbers compare exactly, so a whole x looks only at fractional neighbours.
        width = 5e-7 + 1.1e-5 * abs(x) + 1e-12
        if self._scan(self._fractional if x.is_integer() else self._floats, x - width, x + width, ok):
            return True
        try:
            return self._rounding_match(x, ok)
        except InvalidOperation:
            # Too large to quantize: compare with every float, as the full search did.
            return any(ok(v) for v in self._floats)

    @staticmethod
    def _scan(pool: list[float], lo: float, hi: float, ok: Callable[[object], bool]) -> bool:
        """The first match in [lo, hi], walked in place: a window can hold every value."""
        return any(ok(pool[i]) for i in range(bisect_left(pool, lo), bisect_right(pool, hi)))

    def _rounding_match(self, x: float, ok: Callable[[object], bool]) -> bool:
        exact = Decimal(str(x))
        for places in range(0, 7):
            step = Decimal(1).scaleb(-places)
            # Some value is x rounded to `places`...
            for mode in (ROUND_HALF_EVEN, ROUND_HALF_UP):
                r = float(exact.quantize(step, rounding=mode))
                if self._scan(self._floats, r - 2e-9, r + 2e-9, ok):
                    return True
            # ...or x is some value rounded to `places`, which needs x to have at most that many
            # decimals and puts the value within half a unit of that place. A whole value rounds
            # to itself, so only fractional ones can do this without being equal to x.
            if abs(float(exact.quantize(step, rounding=ROUND_HALF_EVEN)) - x) <= 1e-9:
                half = 0.5 * 10.0 ** -places + 2e-9
                if self._scan(self._fractional, x - half, x + half, ok):
                    return True
        return False


# At and past this magnitude the rounding rule cannot quantize to six places (Decimal keeps 28
# digits), so the got-facts reading raises on such a value against a fraction. Measured: 1e22
# raises against 0.5, 1e21 does not; the bound sits a decade below to be safe.
_UNSURE_MAGNITUDE = 1e21


def _unsure(v: float) -> bool:
    """A float the cell readings meet unpredictably: an infinity matches every number under the
    tolerance (inf <= 1e-5 * inf), and an infinity, a NaN or a huge value can make the rounding
    rule raise. The index compares these outright instead of looking for them in a window."""
    return math.isnan(v) or math.isinf(v) or abs(v) >= _UNSURE_MAGNITUDE


def _distinct(values: list[object]) -> list[object]:
    """Each value once, for the existential checks. Keyed by type as well as value: True and 1.0
    are equal to Python but not to the readings (1.4 rounds to 1.0, not to True)."""
    seen: set[tuple[type, object]] = set()
    out: list[object] = []
    for v in values:
        if v is not None and not _hashable(v):
            out.append(v)
            continue
        key = (type(v), v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _covers(gold_values: list[object], gold_index: _ValueIndex, cand_values: list[object],
            cand_index: _ValueIndex, cell_match: CellMatch) -> bool:
    """Whether every gold value matches some candidate value and every candidate value some
    gold value. A matching choice pairs gold's rows with the candidate's one to one (or, with
    duplicates insignificant, each distinct row with one), so every gold column it uses meets
    this against the candidate column standing in for it."""
    return (all(cand_index.any_match(g, cell_match, x_is_gold=True) for g in gold_values)
            and all(gold_index.any_match(c, cell_match, x_is_gold=False) for c in cand_values))


def _no_rows(gold_rows: list[list[object]], width: int, arity: int) -> Iterator[tuple[int, ...]]:
    """A candidate with no rows matches only a gold with none, and then any choice does."""
    if not gold_rows and arity <= width:
        yield tuple(range(arity))


def _positional_choices(gold_rows: list[list[object]], cand_rows: list[list[object]], arity: int,
                        width: int, cell_match: CellMatch, work: _Work) -> Iterator[tuple[int, ...]]:
    """Increasing column tuples -- the position-wise reading keeps the candidate's column order --
    whose every column covers the gold column it stands in for."""
    if not cand_rows:
        yield from _no_rows(gold_rows, width, arity)
        return
    gold_cols = [_distinct([row[j] for row in gold_rows]) for j in range(arity)]
    cand_cols = [_distinct([row[c] for row in cand_rows]) for c in range(width)]
    gold_idx = [_ValueIndex(col, work) for col in gold_cols]
    cand_idx = [_ValueIndex(col, work) for col in cand_cols]
    fits = [[c for c in range(width)
             if _covers(gold_cols[j], gold_idx[j], cand_cols[c], cand_idx[c], cell_match)]
            for j in range(arity)]
    yield from _increasing_choices(fits)


def _increasing_choices(fits: list[list[int]]) -> Iterator[tuple[int, ...]]:
    """Every increasing tuple taking its j-th column from `fits[j]`, without walking dead ends.

    A plain depth-first walk explores every prefix before finding that a later level has no room
    -- 0/1 flag columns fit most levels, and a 42 that fits none made it walk billions of
    prefixes and yield nothing. So first find the LATEST column each level may take and still
    leave increasing room for the levels after it; a walk held under those bounds completes
    every prefix it starts, and its work is bounded by what it yields. `fits[j]` is ascending.
    """
    latest: list[int] = []
    bound = math.inf
    for options in reversed(fits):
        room = [c for c in options if c < bound]
        if not room:
            return
        bound = max(room)
        latest.append(bound)
    latest.reverse()

    # Iterative, not recursive: a gold of a thousand columns would need a thousand frames.
    k = len(fits)
    if k == 0:
        yield ()
        return
    chosen = [0] * k
    nxt = [0] * k  # where level j resumes in fits[j]
    j = 0
    while j >= 0:
        options = fits[j]
        i = max(nxt[j], bisect_right(options, chosen[j - 1] if j else -1))
        if i < len(options) and options[i] <= latest[j]:
            chosen[j], nxt[j] = options[i], i + 1
            if j == k - 1:
                yield tuple(chosen)
            else:
                j += 1
                nxt[j] = 0
        else:
            j -= 1


def _sorted_choices(gold_rows: list[list[object]], cand_rows: list[list[object]], arity: int,
                    width: int, cell_match: CellMatch, work: _Work) -> Iterator[tuple[int, ...]]:
    """Column sets for the cell-sorted reading. There every candidate cell is compared with SOME
    cell of its gold row, so a usable column holds only values that appear somewhere in gold."""
    if not cand_rows:
        yield from _no_rows(gold_rows, width, arity)
        return
    gold_values = _distinct([cell for row in gold_rows for cell in row])
    gold_cells = _ValueIndex(gold_values, work)
    usable = [c for c in range(width)
              if all(gold_cells.any_match(v, cell_match, x_is_gold=False)
                     for v in _distinct([row[c] for row in cand_rows]))]
    # And every gold cell is compared with some chosen cell, so each must be matchable by a
    # usable column: one gold value nobody holds rules out every set at once.
    usable_cells = _ValueIndex(_distinct([row[c] for row in cand_rows for c in usable]), work)
    if not all(usable_cells.any_match(v, cell_match, x_is_gold=True) for v in gold_values):
        return
    if len(cand_rows) == 1 and len(gold_rows) == 1:
        # One row, the usual shape of a wide answer: the chosen cells pair one to one with
        # gold's, so a set exists only if gold's cells can each take a DIFFERENT usable column.
        # No such pairing rules out every set; one found is the set most likely to pass, so try
        # it first.
        pairing = _pair_cells(gold_rows[0], cand_rows[0], usable, cell_match, work)
        if pairing is None:
            return
        first = tuple(sorted(pairing))
        yield first
        yield from (keep for keep in combinations(usable, arity) if keep != first)
        return
    yield from combinations(usable, arity)


def _pair_cells(gold_row: list[object], cand_row: list[object], usable: list[int],
                cell_match: CellMatch, work: _Work) -> list[int] | None:
    """Give each gold cell a different candidate column holding a matching value, or None.
    Augmenting paths (Kuhn's algorithm): rows are a few dozen cells wide."""
    def edge(g: object, c: object) -> bool:
        work.spend()
        try:
            return cell_match(g, c)
        except ArithmeticError:
            return True  # cannot compare: a possible match, so no set is ruled out on it

    edges = [[c for c in usable if edge(g, cand_row[c])] for g in gold_row]
    owner: dict[int, int] = {}
    if not all(_augment(j, edges, owner) for j in range(len(gold_row))):
        return None
    return list(owner)


def _augment(start: int, edges: list[list[int]], owner: dict[int, int]) -> bool:
    """One augmenting path from gold cell `start`, found with an explicit stack (a path can be
    as long as the row is wide). `path[i]` is the column taken from `stack[i]` to `stack[i+1]`;
    on reaching a free column every cell on the path moves to the column it took."""
    seen: set[int] = set()
    stack = [(start, iter(edges[start]))]
    path: list[tuple[int, int]] = []
    while stack:
        gold, options = stack[-1]
        for c in options:
            if c in seen:
                continue
            seen.add(c)
            path.append((gold, c))
            if c not in owner:
                for g, col in path:
                    owner[col] = g
                return True
            stack.append((owner[c], iter(edges[owner[c]])))
            break
        else:
            stack.pop()
            if path:
                path.pop()
    return False
