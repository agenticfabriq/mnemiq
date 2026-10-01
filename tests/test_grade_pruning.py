"""The got-facts search is pruned, not changed (register M113).

`results_match` used to try every choice of the candidate's columns -- C(candidate, gold) of
them, 4e12 for a 74-column answer to an 11-column gold -- and hung. It now skips choices that
cannot match. These tests hold it to the old verdict: `_brute_force` below is the search as it
was, and the two must agree on every input, including the awkward ones the pruning has to get
right (tolerance edges, roundings in both directions, NULLs, bools, duplicates, NaN).
"""
from __future__ import annotations

import random
from itertools import combinations

import pyarrow as pa
import pytest

from mnemiq.eval.grade import GotFactsUndecided, _distinct_rows, results_match
from mnemiq.execute.resultset import _match_rows, _rows, facts_cells_match, grade_cells_match


def _brute_force(gold: pa.Table, candidate: pa.Table, allow_extra_columns: bool = True,
                 duplicate_rows_insignificant: bool = False, optimistic: bool = False) -> bool:
    """The search before pruning, kept verbatim as the oracle. `optimistic` reads a pair the
    cell readings cannot compare (they raise) as a match: the most any resolution could match."""
    cell_match = facts_cells_match if allow_extra_columns else grade_cells_match
    if optimistic:
        strict_match = cell_match

        def cell_match(a, b):
            try:
                return strict_match(a, b)
            except ArithmeticError:
                return True
    if allow_extra_columns:
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
    all_rows = _rows(candidate)
    all_keys = [[repr(cell) for cell in row] for row in all_rows]
    sorted_gold: list[list[object]] = []
    if allow_extra_columns:
        sorted_gold = [sorted(row, key=repr) for row in gold_rows]
        if duplicate_rows_insignificant:
            sorted_gold = _distinct_rows(sorted_gold)
    for keep in column_choices:
        candidate_rows = [[row[i] for i in keep] for row in all_rows]
        if duplicate_rows_insignificant:
            candidate_rows = _distinct_rows(candidate_rows)
        if _match_rows(gold_rows, candidate_rows, 0.0, cell_match):
            return True
        if not allow_extra_columns:
            continue
        sorted_candidate = [
            [value for _, value in sorted(((keys[i], row[i]) for i in keep), key=lambda p: p[0])]
            for row, keys in zip(all_rows, all_keys, strict=True)
        ]
        if duplicate_rows_insignificant:
            sorted_candidate = _distinct_rows(sorted_candidate)
        if _match_rows(sorted_gold, sorted_candidate, 0.0, cell_match):
            return True
    return False


# Values chosen to sit on the edges the pruning must not cross: a rounding either way, a value
# just inside and just outside the numeric tolerance, integers that must stay exact, NULL and
# NaN. Each column holds ONE type, so arrow keeps it -- a mixed column would be stringified and
# turn every tolerance case into string equality.
_FLOATS = [0.0, 1.0, 2.0, 1.4249, 1.42, 1.425, 1.43, 1.4, 1.0000004, 1.000006, 52.63, 53.0, 0.196,
           0.2, 38.125, 38.12, 38.13, 7.0, -1.5, -2.0, None, float("nan"),
           # Within the 1e-5 relative tolerance of each other but no rounding of each other --
           # the only pairs that test the tolerance window on its own -- and one just outside.
           1234567.5, 1234570.25, 1234599.5,
           # Large WHOLE numbers: inside 1e-5 of each other, yet counts compare exactly.
           1e6, 1e6 + 5.0, 1e6 + 20.0]
# The readings' own edges, drawn rarely: an infinity meets every number under the tolerance
# (inf <= 1e-5 * inf), and inf against inf, or 1e22 against a fraction, raises inside the
# rounding rule -- the full search crashes there, which the oracle allows for.
_EDGES = [float("inf"), float("-inf"), 1e22]
_STRINGS = ["a", "b", "1", "1.0", None]
_BOOLS = [True, False, None]


def _col(rng: random.Random, n: int) -> list[object]:
    pool = rng.choices([_FLOATS, _STRINGS, _BOOLS], [6, 2, 1])[0]

    def pick() -> object:
        return rng.choice(_EDGES) if pool is _FLOATS and rng.random() < 0.02 else rng.choice(pool)

    if rng.random() < 0.15:  # a constant column: many choices look alike
        return [pick()] * n
    return [pick() for _ in range(n)]


def _as_planted(rng: random.Random, col: list[object]) -> list[object]:
    """Gold's column as a candidate might spell it: rounded, or a bool read as 1.0/0.0
    (Python calls True == 1.0, and both cell readings follow it)."""
    if all(v is None or isinstance(v, bool) for v in col) and rng.random() < 0.5:
        return [None if v is None else float(v) for v in col]
    if rng.random() < 0.3:
        return [round(v, 2) if isinstance(v, float) and v == v else v for v in col]
    return list(col)


def _table(cols: list[list[object]]) -> pa.Table:
    return pa.Table.from_arrays([pa.array(c) for c in cols], names=[f"c{i}" for i in range(len(cols))])


def _case(rng: random.Random, extra_columns: bool) -> tuple[pa.Table, pa.Table]:
    rows = rng.choice([0, 1, 1, 2, 3, 4])
    gold_cols = [_col(rng, rows) for _ in range(rng.choice([0, 1, 1, 2, 2, 3]))]
    gold = _table(gold_cols) if gold_cols else pa.table({})
    # Exact match needs the same width, so there the candidate is mostly gold's own columns.
    width = rng.choice([1, 2, 3, 4, 5, 6]) if extra_columns or rng.random() < 0.3 else 0
    cand_cols = [_col(rng, rows) for _ in range(width)]
    # Often plant gold's columns in the candidate -- respelled, reordered, rows shuffled -- so
    # that matches are common enough to test the True side too.
    if gold_cols and rng.random() < 0.7:
        planted = [_as_planted(rng, c) for c in gold_cols]
        slots = sorted(rng.sample(range(width + len(planted)), len(planted)))
        if rng.random() < 0.3:
            rng.shuffle(slots)
        for slot, col in zip(slots, planted, strict=True):
            cand_cols.insert(min(slot, len(cand_cols)), col)
        if rows > 1 and rng.random() < 0.4:  # the same rows in a different order
            order = list(range(rows))
            rng.shuffle(order)
            cand_cols = [[c[i] for i in order] for c in cand_cols]
    return gold, _table(cand_cols)


@pytest.mark.parametrize("allow_extra", [True, False])
@pytest.mark.parametrize("dupes", [False, True])
def test_the_pruned_search_agrees_with_the_full_one(allow_extra, dupes):
    rng = random.Random(1113 + 2 * allow_extra + dupes)
    agree = matched = crashed = 0
    for _ in range(4000):
        gold, cand = _case(rng, allow_extra)
        try:
            expected = _brute_force(gold, cand, allow_extra, dupes)
        except ArithmeticError:
            # The full search crashed on a pair the readings cannot compare. The pruned one may
            # answer True (a choice matched outright) or undecided; it may answer False only if
            # nothing could match even reading every such pair as a match.
            crashed += 1
            try:
                got = results_match(gold, cand, allow_extra, dupes)
            except GotFactsUndecided:
                got = None
            if got is False:
                assert not _brute_force(gold, cand, allow_extra, dupes, optimistic=True), (
                    gold.to_pydict(), cand.to_pydict(), allow_extra, dupes)
            continue
        got = results_match(gold, cand, allow_extra, dupes)
        assert got == expected, (gold.to_pydict(), cand.to_pydict(), allow_extra, dupes)
        agree += 1
        matched += expected
    assert crashed < agree * 0.1, f"{crashed} crashes leave too little compared"
    # A sampler that never produced a match would test only the False side.
    assert matched > agree * 0.1, f"only {matched} of {agree} cases matched"


@pytest.mark.parametrize("dupes", [False, True])
@pytest.mark.parametrize("edge", [float("inf"), float("-inf")])
def test_an_infinite_gold_value_is_still_matched_as_the_full_search_matches_it(edge, dupes):
    """Found by review: the tolerance reads inf <= 1e-5 * inf as true, so an infinite gold value
    matches ANY finite one, and the windows around a finite candidate never reached it. Whether
    that rule is right is a separate question; the pruning must not change it."""
    gold, cand = pa.table({"g": [edge]}), pa.table({"c": [1.5], "x": ["pad"]})
    assert _brute_force(gold, cand, True, dupes) is True
    assert results_match(gold, cand, True, dupes) is True


def test_a_pair_the_readings_cannot_compare_is_undecided_never_wrong():
    """inf against inf raises inside the rounding rule (inf - inf is NaN, then Decimal cannot
    quantize Infinity), so the full search crashes here. Found by review: skipping the choice
    turned an answer identical to gold into a silent WRONG. It is undecided."""
    gold, cand = pa.table({"g": [float("inf")]}), pa.table({"c": [float("inf")]})
    with pytest.raises(ArithmeticError):
        _brute_force(gold, cand)
    with pytest.raises(GotFactsUndecided, match="cannot compare"):
        results_match(gold, cand)


def test_a_match_elsewhere_still_decides_past_an_uncomparable_pair():
    """2.5 against inf raises inside the rounding rule, and the full search meets that column
    first and crashes. The next column matches outright, so the answer is decided: True."""
    gold = pa.table({"g": [2.5]})
    cand = pa.table({"a": [float("inf")], "b": [2.5]})
    with pytest.raises(ArithmeticError):
        _brute_force(gold, cand)
    assert results_match(gold, cand) is True


def test_a_long_column_of_near_equal_values_is_read_once_per_value(monkeypatch):
    """Found by review: 10,000 rows of 1000.1 against 10,000 of 1000.1001 cost 2N queries that
    each copied an N-value window before looking. Each distinct value is now read once and a
    window is walked only to its first match."""
    import mnemiq.eval.grade as grade

    calls = [0]
    real = grade.facts_cells_match

    def counted(a, b):
        calls[0] += 1
        return real(a, b)

    monkeypatch.setattr(grade, "facts_cells_match", counted)
    n = 10_000
    gold = pa.table({"g": [1000.1] * n})
    cand = pa.table({"c": [1000.1001] * n, "x": [7.25] * n})
    assert results_match(gold, cand)
    assert calls[0] < 100 + 2 * n, calls[0]  # the pruning reads 2 values; the check compares n rows


class _NoSlicing(list):
    """A pool that counts the items read and refuses to be copied."""

    reads = 0

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise AssertionError("the window was copied")
        _NoSlicing.reads += 1
        return super().__getitem__(i)


def test_a_window_is_walked_in_place_and_only_to_its_first_match():
    from mnemiq.eval.grade import _ValueIndex

    _NoSlicing.reads = 0
    pool = _NoSlicing([1.0 + i * 1e-9 for i in range(5000)])
    assert _ValueIndex._scan(pool, 0.0, 2.0, lambda v: True)
    # two binary searches find the window's ends; then one value is read, not 5,000
    assert _NoSlicing.reads <= 2 * len(pool).bit_length() + 1


def test_pruning_that_runs_out_of_budget_says_it_cannot_decide(monkeypatch):
    import mnemiq.eval.grade as grade

    gold = pa.table({"g": [float(i) + 0.5 for i in range(50)]})
    cand = pa.table({f"x{j}": [float(i) + 0.25 for i in range(50)] for j in range(5)})
    assert not results_match(gold, cand)  # control: decided within the real budget
    monkeypatch.setattr(grade, "MAX_PRUNING_STEPS", 3)
    with pytest.raises(GotFactsUndecided, match="lookups and cell comparisons"):
        results_match(gold, cand)


def _counting(monkeypatch) -> list[int]:
    """How many column choices reach the row comparison: the work, counted rather than timed."""
    import mnemiq.eval.grade as grade

    calls = [0]
    real = grade._match_rows

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(grade, "_match_rows", counted)
    return calls


def test_a_wide_answer_holding_every_gold_column_is_graded(monkeypatch):
    """The shape that hung: one row, gold's 11 columns among 74. C(74, 11) is 4e12 choices."""
    calls = _counting(monkeypatch)
    rng = random.Random(7)
    gold_vals = {f"g{i}": [round(rng.uniform(0, 100), 4)] for i in range(11)}
    cand = {f"x{i}": [round(rng.uniform(100, 200), 4)] for i in range(63)}
    cand.update({f"c_{k}": v for k, v in gold_vals.items()})
    assert results_match(pa.table(gold_vals), pa.table(cand), allow_extra_columns=True)
    assert calls[0] <= 2


def test_a_wide_wrong_answer_is_refused_without_trying_choices(monkeypatch):
    """And the False side, where the old search had to try every choice before giving up."""
    calls = _counting(monkeypatch)
    gold = pa.table({f"g{i}": [float(i) + 0.5] for i in range(11)})
    cand = pa.table({f"x{i}": [float(i) + 0.25] for i in range(74)})
    assert not results_match(gold, cand, allow_extra_columns=True)
    assert calls[0] == 0


def test_flag_columns_cannot_hide_a_missing_gold_value(monkeypatch):
    """Found by review: 0/1/NULL columns appear in almost any gold row, so per-column pruning
    keeps all 74 -- and with gold's 42 held by none of them, every set is still wrong. One gold
    value no usable column holds rules them all out at once."""
    calls = _counting(monkeypatch)
    flags = [0.0, 1.0, None]
    gold = pa.table({f"g{i}": [v] for i, v in enumerate(
        [0.0, 1.0, None, 0.0, 1.0, None, 0.0, 1.0, 0.0, 1.0, 42.0])})
    cand = pa.table({f"x{i}": pa.array([flags[i % 3]], pa.float64()) for i in range(74)})
    assert not results_match(gold, cand, allow_extra_columns=True)
    assert calls[0] == 0


def test_a_gold_value_no_column_holds_rules_out_every_set_on_many_rows_too(monkeypatch):
    """On one row the cell pairing already catches it; on two the held-value check is what
    stops the cell-sorted reading from trying every set of three flag columns."""
    calls = _counting(monkeypatch)
    gold = pa.table({"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [42.0, 0.0]})
    cand = pa.table({f"x{i}": [[0.0, 1.0][i % 2], [1.0, 0.0][i % 2]] for i in range(12)})
    assert not results_match(gold, cand, allow_extra_columns=True)
    assert calls[0] == 0


class _Budget(list):
    """A list that counts how often it is walked, and stops a walk that has run away."""

    steps = 0

    def __iter__(self):
        _Budget.steps += 1
        if _Budget.steps > 10_000:
            raise AssertionError("the walk is exploring dead ends")
        return super().__iter__()


@pytest.mark.parametrize("last", [[], [5]], ids=["no-column-fits", "no-room-left"])
def test_the_position_wise_walk_does_not_explore_dead_ends(last):
    """Found by timing the flag test: ten levels that fit 60 columns each, and a last level that
    fits none (or only a column too early to follow ten others). A plain walk explores C(60, 10)
    prefixes before it learns that; this one yields nothing without walking them."""
    from mnemiq.eval.grade import _increasing_choices

    _Budget.steps = 0
    fits = [_Budget(range(60)) for _ in range(10)] + [_Budget(last)]
    assert list(_increasing_choices(fits)) == []


def test_the_position_wise_walk_stays_inside_the_room_it_has():
    """The bounds, not the up-front check: the last level fits only column 10, so the ten before
    it must be 0..9 -- exactly one tuple. Unbounded, the walk tries prefixes up to column 59 and
    abandons each one."""
    from mnemiq.eval.grade import _increasing_choices

    _Budget.steps = 0
    fits = [_Budget(range(60)) for _ in range(10)] + [_Budget([10])]
    assert list(_increasing_choices(fits)) == [tuple(range(11))]


def test_the_position_wise_walk_yields_exactly_the_increasing_tuples():
    from mnemiq.eval.grade import _increasing_choices

    rng = random.Random(3)
    for _ in range(300):
        width, arity = rng.randint(0, 7), rng.randint(0, 4)
        fits = [sorted(rng.sample(range(width), rng.randint(0, width))) for _ in range(arity)]
        expected = [t for t in combinations(range(width), arity)
                    if all(c in fits[j] for j, c in enumerate(t))]
        assert list(_increasing_choices(fits)) == expected


def test_one_row_flags_in_the_wrong_counts_are_refused_without_trying_choices(monkeypatch):
    """The same 74 flag columns with a 42 among them: every gold value is held, but gold needs
    four zeros and the row has two. Gold's cells cannot each take a different column, so no set
    of eleven can match."""
    calls = _counting(monkeypatch)
    gold = pa.table({f"g{i}": [v] for i, v in enumerate(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, None, None, 1.0, 1.0, 42.0])})
    cols = {f"x{i}": pa.array([v], pa.float64()) for i, v in enumerate([0.0, 0.0, 42.0])}
    cols.update({f"y{i}": pa.array([[1.0, None][i % 2]], pa.float64()) for i in range(71)})
    assert not results_match(gold, pa.table(cols), allow_extra_columns=True)
    assert calls[0] == 0


def test_a_wide_answer_under_the_cell_sorted_reading_is_graded():
    """Gold's columns present but in a different order: only the cell-sorted retry finds it."""
    gold = pa.table({"k": ["a", "b"], "n": [3.0, 5.0]})
    cand = {f"x{i}": [f"z{i}", f"y{i}"] for i in range(40)}
    cand.update({"count": [3.0, 5.0], "key": ["a", "b"]})
    assert results_match(gold, pa.table(cand), allow_extra_columns=True)


def _undecidable() -> tuple[pa.Table, pa.Table]:
    """Two rows of flags: every column survives pruning and no set of three matches."""
    gold = pa.table({"a": [0.0, 1.0], "b": [1.0, 0.0], "c": [0.0, 0.0]})
    cand = pa.table({f"x{i}": [[0.0, 1.0][i % 2], [1.0, 0.0][i % 2]] for i in range(12)})
    return gold, cand


def test_past_the_limit_got_facts_says_it_cannot_decide(monkeypatch):
    import mnemiq.eval.grade as grade

    gold, cand = _undecidable()
    assert not results_match(gold, cand)  # control: decided (False) at the real limit
    monkeypatch.setattr(grade, "MAX_CHOICES", 5)
    with pytest.raises(GotFactsUndecided):
        results_match(gold, cand)


def test_the_harness_records_undecided_as_an_error_not_a_wrong_answer(monkeypatch):
    """The join: run_case's own grading, not results_match alone. WRONG is the outcome nobody
    sees; an undecided grade must surface."""
    import mnemiq.eval.grade as grade
    from mnemiq.eval.harness import Outcome, run_case
    from test_harness import _answered, _case

    gold, cand = _undecidable()

    class _Adapter:
        def execute_arrow(self, sql, timeout_s=30):
            return cand if sql == "CANDIDATE" else gold

    monkeypatch.setattr(grade, "MAX_CHOICES", 5)
    result = run_case(_case(), lambda q: _answered(), _Adapter())
    assert result.outcome == Outcome.ERROR and "undecided" in result.answer


def test_spider2_records_undecided_as_an_error_with_its_reason(monkeypatch):
    """The join again, through run_case_csv: the reason reaches the result, not only ERROR."""
    import mnemiq.eval.grade as grade
    from mnemiq.agent.loop import AgentAnswer
    from mnemiq.eval.harness import Outcome
    from mnemiq.eval.spider2 import grade_alternatives, run_case_csv
    from test_spider2 import _Adapter, _case, _trace

    gold, cand = _undecidable()
    monkeypatch.setattr(grade, "MAX_CHOICES", 5)
    engine = lambda q: AgentAnswer(answer="flags", trace=_trace("SELECT *"))  # noqa: E731
    result = run_case_csv(_case(), engine, _Adapter(cand), [gold])
    assert result.outcome == Outcome.ERROR and "got-facts undecided" in result.answer
    decides = pa.table({"x0": [0.0, 1.0]})  # cand's first column: got-facts, decided
    assert grade_alternatives(cand, [gold, decides]) == Outcome.CORRECT_FACTS
