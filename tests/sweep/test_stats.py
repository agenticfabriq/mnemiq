"""The statistics a sweep decides by: the exact McNemar test, Holm's correction, and the smallest gain they can show."""

from __future__ import annotations

import pytest

from mnemiq.sweep.stats import detectable_gain, holm, mcnemar_p


def test_mcnemar_matches_the_exact_binomial():
    assert mcnemar_p(6, 0) == pytest.approx(0.03125) and mcnemar_p(3, 1) == pytest.approx(0.625)


def test_holm_raises_the_bar_with_the_number_of_arms():
    assert holm({"a": 0.03}, 0.05) == {"a"}
    assert holm({"a": 0.03, **{f"x{i}": 0.9 for i in range(9)}}, 0.05) == set()


def test_detectable_gain_grows_with_the_arms_compared():
    assert [detectable_gain(500, m, 0.05) for m in (1, 5, 10, 20)] == [6, 8, 9, 10]
    assert detectable_gain(8, 10, 0.05) is None
