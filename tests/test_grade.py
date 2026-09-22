from datetime import datetime
from decimal import Decimal

import pyarrow as pa

from mnemiq.eval.grade import results_match


def _t(data: dict) -> pa.Table:
    return pa.table(data)


def test_an_identical_result_matches():
    assert results_match(_t({"n": [820]}), _t({"n": [820]}))


def test_a_different_column_name_still_matches():
    # the model aliases freely; grading names would be string matching by the back door
    assert results_match(_t({"n": [820]}), _t({"total_claims": [820]}))


def test_a_different_row_order_still_matches():
    gold = _t({"k": ["no", "yes"], "n": [128, 692]})
    candidate = _t({"k": ["yes", "no"], "n": [692, 128]})
    assert results_match(gold, candidate)


def test_a_different_column_order_still_matches():
    gold = _t({"k": ["yes"], "n": [692]})
    candidate = _t({"n": [692], "k": ["yes"]})
    assert results_match(gold, candidate)


def test_a_wrong_number_fails():
    assert not results_match(_t({"n": [820]}), _t({"n": [819]}))


def test_a_missing_row_fails():
    gold = _t({"k": ["no", "yes"], "n": [128, 692]})
    candidate = _t({"k": ["yes"], "n": [692]})
    assert not results_match(gold, candidate)


def test_presentation_rounding_is_tolerated():
    # 0.6429 -> 0.64 is a formatting choice, not a computational error
    assert results_match(_t({"r": [0.6429268]}), _t({"r": [0.6429]}))


def test_a_materially_different_number_is_not_tolerated():
    # a wrong filter or a missing DISTINCT moves a number far more than 1%
    assert not results_match(_t({"r": [0.6429]}), _t({"r": [0.58]}))


def test_decimal_and_float_are_the_same_number():
    assert results_match(_t({"total": [Decimal("13600.00")]}), _t({"total": [13600.0]}))


def test_an_integer_and_its_float_are_the_same_number():
    assert results_match(_t({"n": [820]}), _t({"n": [820.0]}))


def test_dates_compare_by_value():
    gold = _t({"d": [datetime(2019, 1, 15)]})
    assert results_match(gold, _t({"d": [datetime(2019, 1, 15)]}))
    assert not results_match(gold, _t({"d": [datetime(2019, 1, 16)]}))


def test_an_empty_result_matches_an_empty_result():
    empty = pa.table({"n": pa.array([], type=pa.int64())})
    assert results_match(empty, empty)


def test_an_empty_result_does_not_match_a_populated_one():
    empty = pa.table({"n": pa.array([], type=pa.int64())})
    assert not results_match(empty, _t({"n": [1]}))


def test_a_null_matches_a_null():
    nulls = pa.table({"n": pa.array([None], type=pa.int64())})
    assert results_match(nulls, nulls)
    assert not results_match(nulls, _t({"n": [0]}))  # NULL is not zero


def test_an_extra_context_column_does_not_make_a_right_answer_wrong():
    # "which policy is earliest?" answered with the policy number AND its date is not an
    # error -- the fact asked for is there. Found live: three of four eval "failures" were
    # right answers carrying one extra column.
    gold = _t({"policy_number": ["31003000336"]})
    candidate = _t({"policy_number": ["31003000336"], "effective_date": ["2015-01-01"]})
    assert results_match(gold, candidate)


def test_extra_columns_cannot_rescue_wrong_rows():
    gold = _t({"n": [820]})
    candidate = _t({"n": [819], "extra": ["context"]})
    assert not results_match(gold, candidate)


def test_extra_columns_cannot_rescue_a_missing_row():
    gold = _t({"d": ["2019-01-15", "2019-06-02"]})
    candidate = _t({"id": [1], "d": ["2019-01-15"]})
    assert not results_match(gold, candidate)


def test_strict_mode_rejects_an_extra_column():
    gold = _t({"n": [820]})
    candidate = _t({"n": [820], "extra": ["ctx"]})
    assert results_match(gold, candidate)  # lenient default: still correct
    assert not results_match(gold, candidate, allow_extra_columns=False)  # BIRD: wrong


def test_strict_mode_ignores_row_order_but_not_column_order():
    # This used to assert that strict ignored BOTH, which made mnemiq's BIRD "correct"
    # generous against the published metric: BIRD's evaluator compares row tuples
    # position-wise. Row order stays tolerated; column order moved to got-facts only.
    # See beacon docs/grading.md, "exact".
    gold = _t({"k": ["no", "yes"], "n": [128, 692]})

    rows_reordered = _t({"k": ["yes", "no"], "n": [692, 128]})
    assert results_match(gold, rows_reordered, allow_extra_columns=False)

    columns_reordered = _t({"n": [692, 128], "k": ["yes", "no"]})
    assert not results_match(gold, columns_reordered, allow_extra_columns=False)
    assert results_match(gold, columns_reordered, allow_extra_columns=True)


def test_the_gold_may_never_have_more_columns_than_the_candidate():
    # the tolerance is one-directional: the candidate may add context, never omit facts
    gold = _t({"k": ["yes"], "n": [692]})
    candidate = _t({"n": [692]})
    assert not results_match(gold, candidate)


def test_got_facts_accepts_a_rounding_of_the_same_quantity():
    # 66.62 for 66.6230 is the same quantity printed shorter -- the fact is there.
    gold = pa.table({"rate": [66.6230]})
    candidate = pa.table({"rate": [66.62]})

    assert results_match(gold, candidate, allow_extra_columns=True)


def test_exact_match_does_not_accept_that_rounding():
    # Exact match claims the result sets are the SAME; presentation leniency belongs to
    # the second metric, or the strict number stops meaning what BIRD reports.
    gold = pa.table({"rate": [66.6230]})
    candidate = pa.table({"rate": [66.62]})

    assert not results_match(gold, candidate, allow_extra_columns=False)


def test_a_rounding_is_read_in_either_direction():
    assert results_match(pa.table({"r": [52.63]}), pa.table({"r": [52.6]}),
                         allow_extra_columns=True)
    assert results_match(pa.table({"r": [52.6]}), pa.table({"r": [52.63]}),
                         allow_extra_columns=True)


def test_rounding_a_small_quantity_to_zero_is_not_a_fact():
    # 0.0 for 0.196 collapses the quantity to nothing; that answer lost the fact, so
    # rounding to zero places is presentation only above magnitude one.
    assert not results_match(pa.table({"r": [0.196]}), pa.table({"r": [0.0]}),
                             allow_extra_columns=True)
    assert results_match(pa.table({"r": [52.63]}), pa.table({"r": [53.0]}),
                         allow_extra_columns=True)


def test_a_different_quantity_nearby_is_still_wrong():
    # The reason this is a rounding rule and not a wider band: 1038.15 is not 1039.32
    # printed differently, it is a different number, and a percentage would admit it.
    assert not results_match(pa.table({"n": [1039.324324]}),
                             pa.table({"n": [1038.150684931507]}),
                             allow_extra_columns=True)


def test_counts_stay_exact_under_the_tolerant_reading_too():
    assert not results_match(pa.table({"n": [819]}), pa.table({"n": [820]}),
                             allow_extra_columns=True)


def test_either_rounding_convention_is_still_a_rounding():
    # 38.125 to two places is 38.12 under banker's rounding and 38.13 under half-up.
    # Both are the same quantity printed shorter; which convention the engine used is
    # not the candidate's answer. (Spider local023 was failing on exactly this pair.)
    gold = pa.table({"avg": [38.125]})

    assert results_match(gold, pa.table({"avg": [38.13]}), allow_extra_columns=True)
    assert results_match(gold, pa.table({"avg": [38.12]}), allow_extra_columns=True)


def test_a_neighbouring_value_is_not_rescued_by_either_convention():
    # 6.47 is not 6.48 rounded any way at any place -- it is a different average.
    assert not results_match(pa.table({"r": [6.48]}), pa.table({"r": [6.47]}),
                             allow_extra_columns=True)


def test_column_order_is_tolerated_by_got_facts():
    # SELECT k, count(*) and SELECT count(*), k carry the same information.
    gold = pa.table({"k": ["a", "b"], "n": [1, 2]})
    swapped = pa.table({"n": [1, 2], "k": ["a", "b"]})

    assert results_match(gold, swapped, allow_extra_columns=True)


def test_column_order_is_part_of_exact_match():
    # BIRD's evaluator compares row tuples position-wise, so the strict number has to
    # stay the one the leaderboard publishes. beacon docs/grading.md, "exact".
    gold = pa.table({"k": ["a", "b"], "n": [1, 2]})
    swapped = pa.table({"n": [1, 2], "k": ["a", "b"]})

    assert not results_match(gold, swapped, allow_extra_columns=False)


def test_exact_match_still_ignores_row_order():
    # Row order is a separate question from column order, and stays tolerated here.
    gold = pa.table({"n": [1, 2]})

    assert results_match(gold, pa.table({"n": [2, 1]}), allow_extra_columns=False)


# ---------------------------------------------------------------------------
# M105: the strict reading refused on row MULTIPLICITY where BIRD's own rule
# collapses duplicates -- a violation of the shared grading contract, not a
# product choice. beacon's docs/grading.md has BIRD items declare
# `duplicate_rows_insignificant` "so that exact on BIRD is the number the
# leaderboard publishes", measured there at 24 cases in 2,899 (0.83 points).
# ---------------------------------------------------------------------------


def _dupes():
    """Same distinct rows, different multiplicity -- wrong under a multiset rule, right
    under `set(pred) == set(gold)`."""
    return pa.table({"n": [1, 1, 2]}), pa.table({"n": [1, 2]})


def test_duplicates_still_differ_when_the_benchmark_does_not_declare_it():
    """The DEFAULT is unchanged, and deliberately so: absent a declaration, the same rows at
    a different multiplicity are not obviously the same answer."""
    gold, cand = _dupes()
    assert not results_match(gold, cand, allow_extra_columns=False)
    assert not results_match(gold, cand, allow_extra_columns=True)


def test_a_benchmark_that_declares_it_grades_the_way_its_leaderboard_does():
    gold, cand = _dupes()
    assert results_match(gold, cand, allow_extra_columns=False,
                         duplicate_rows_insignificant=True)


def test_duplicates_collapse_on_the_CANDIDATE_side_too():
    """beacon's contract says duplicates collapse "in either direction", and a fixture with
    the repetition only on the gold side never reaches the candidate branch.

    Caught by mutation: disabling the candidate-side collapse left every other test in this
    file green, because `_dupes()` puts the duplicate rows in gold alone.
    """
    gold, cand = pa.table({"n": [1, 2]}), pa.table({"n": [1, 1, 2]})
    assert not results_match(gold, cand, allow_extra_columns=False)
    assert results_match(gold, cand, allow_extra_columns=False,
                         duplicate_rows_insignificant=True)


def test_the_collapse_reaches_both_metrics_as_the_contract_says():
    """beacon honours the flag "in both metrics", so got-facts collapses too. Pinning only
    the exact path would let the two repos diverge on the got-facts column."""
    gold, cand = _dupes()
    assert results_match(gold, cand, allow_extra_columns=True,
                         duplicate_rows_insignificant=True)


def test_collapsing_duplicates_does_not_make_a_wrong_answer_right():
    """The flag removes multiplicity from the comparison and nothing else. A candidate
    missing one of the DISTINCT rows is still wrong, which is what stops this being a
    blanket loosening."""
    gold = pa.table({"n": [1, 1, 2]})
    assert not results_match(gold, pa.table({"n": [1, 1]}), allow_extra_columns=False,
                             duplicate_rows_insignificant=True)
    assert not results_match(gold, pa.table({"n": [1, 3]}), allow_extra_columns=False,
                             duplicate_rows_insignificant=True)


def test_the_engines_own_comparator_is_untouched():
    """THE POINT OF PUTTING THE FLAG IN results_match AND NOT IN _match_rows.

    `results_equal` and `cluster` share `_match_rows` with grading, and `agent/loop.py` votes
    with `cluster`. A dedupe added inside `_match_rows` would change which candidates count as
    agreeing -- every grading test would stay green while the engine silently picked a
    different answer. `grade_cells_match`'s docstring warns about exactly this coupling.
    """
    from mnemiq.execute.resultset import results_equal

    gold, cand = _dupes()
    assert not results_equal(gold, cand), "self-consistency clustering started collapsing rows"


def test_the_cell_sorted_retry_recollapses_what_it_makes_identical():
    """The got-facts branch canonicalises each row's cell order and can turn two distinct
    rows into one, so a dedupe done only before the loop leaves duplicates this branch
    created -- and "collapses in BOTH metrics" then fails on exactly this path.

    Gold states one fact in two column orders; the candidate states it once. Column order is
    presentation to got-facts, so after sorting they are the same single row.
    """
    gold = pa.table({"a": [1, 2], "b": [2, 1]})
    cand = pa.table({"a": [1], "b": [2]})
    assert not results_match(gold, cand, allow_extra_columns=True), "control: no declaration"
    assert results_match(gold, cand, allow_extra_columns=True,
                         duplicate_rows_insignificant=True)


def test_each_cell_is_normalized_once_not_once_per_column_projection(monkeypatch):
    # The column choices are combinatorial -- C(candidate_cols, gold_cols) -- and the candidate
    # used to be re-read, and so re-normalized, inside that loop. This asserts the call COUNT
    # rather than a duration: the redundancy grows without bound in the column count while the
    # wall-clock gain does not, so a timing assertion would measure the wrong thing and flake.
    import mnemiq.execute.resultset as resultset

    calls = 0
    real = resultset.normalize

    def counting(value):
        nonlocal calls
        calls += 1
        return real(value)

    monkeypatch.setattr(resultset, "normalize", counting)

    rows, cand_cols, gold_cols = 20, 8, 3
    candidate = _t({f"c{i}": list(range(rows)) for i in range(cand_cols)})
    gold = _t({f"g{i}": [v + 1000 for v in range(rows)] for i in range(gold_cols)})

    assert not results_match(gold, candidate)  # no projection matches, so ALL are tried
    assert calls == rows * (cand_cols + gold_cols)


def test_a_repeated_column_is_projected_from_its_own_values(monkeypatch):
    # Index projection is only equal to select() projection because _rows reads positionally.
    # If _rows ever goes back through a name-keyed dict, the duplicated name collapses and this
    # candidate stops carrying a 1 in the column the gold wants.
    candidate = pa.Table.from_arrays(
        [pa.array([1]), pa.array([2]), pa.array([3])], names=["a", "a", "b"]
    )
    assert results_match(_t({"g": [1]}), candidate)
    assert results_match(_t({"g": [2]}), candidate)
