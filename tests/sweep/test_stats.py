"""The statistics a sweep decides by: the exact McNemar test, Holm's correction, and the smallest gain they can show."""

from __future__ import annotations

import pytest

from mnemiq.sweep.stats import detectable_gain, holm, mcnemar_p, mcnemar_p_one_sided


def test_mcnemar_matches_the_exact_binomial():
    assert mcnemar_p(6, 0) == pytest.approx(0.03125) and mcnemar_p(3, 1) == pytest.approx(0.625)


def test_the_one_sided_p_counts_the_tail_on_one_side_only():
    # 5 of 7 discordant questions to one side: at least that many by chance is 29 in 128, and half the two-sided p.
    assert mcnemar_p_one_sided(4, 0) == pytest.approx(1 / 16) and mcnemar_p_one_sided(5, 2) == pytest.approx(29 / 128)
    assert mcnemar_p_one_sided(5, 2) == pytest.approx(mcnemar_p(5, 2) / 2)
    assert mcnemar_p_one_sided(1, 3) == pytest.approx(15 / 16) and mcnemar_p_one_sided(0, 0) == 1.0


def test_holm_raises_the_bar_with_the_number_of_arms():
    assert holm({"a": 0.03}, 0.05) == {"a"}
    assert holm({"a": 0.03, **{f"x{i}": 0.9 for i in range(9)}}, 0.05) == set()


def test_holm_lowers_the_bar_after_each_pass():
    # The first p is held to alpha / 2 and the second to alpha / 1; holding both to alpha / 2 would drop b.
    assert holm({"a": 0.01, "b": 0.04}, 0.05) == {"a", "b"}


def test_holm_needs_each_p_strictly_below_its_bar():
    assert holm({"a": 0.05}, 0.05) == set()
    assert holm({"a": 0.01, "b": 0.05}, 0.05) == {"a"}


def test_detectable_gain_grows_with_the_arms_compared():
    assert [detectable_gain(500, m, 0.05) for m in (1, 5, 10, 20)] == [6, 8, 9, 10]
    assert detectable_gain(8, 10, 0.05) is None
