"""The statistics a sweep decides by.

An arm is compared with the start question by question. Only the discordant questions count: those one of the two got
right and the other did not. The exact two-sided McNemar test asks how likely a split at least that uneven would be if
either side were equally likely to win each of them. Every arm compared with the start is one more chance of a lucky
win, so Holm's step-down correction raises the bar with the number of arms.

The screen asks a narrower question of each value it tries, whether it helped, so it reads one tail only: how likely the
arm would win at least that many of the discordant questions by chance.
"""

from __future__ import annotations

import math
from fractions import Fraction


def mcnemar_p(only: int, other: int) -> float:
    """The exact two-sided p of `only` questions right for one side alone and `other` for the other side alone."""
    if only < 0 or other < 0:
        raise ValueError("a count of questions cannot be negative")
    n = only + other
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only, other) + 1))
    return float(min(Fraction(2 * tail, 2 ** n), Fraction(1)))


def mcnemar_p_one_sided(only: int, other: int) -> float:
    """The exact one-sided p that the side right alone on `only` questions, against `other` for the other side, wins at
    least that many of them by chance."""
    if only < 0 or other < 0:
        raise ValueError("a count of questions cannot be negative")
    n = only + other
    if n == 0:
        return 1.0
    return float(Fraction(sum(math.comb(n, i) for i in range(only, n + 1)), 2 ** n))


def holm(pvalues: dict[str, float], alpha: float) -> set[str]:
    """The names whose p passes Holm's step-down correction at `alpha`: the smallest p must be below alpha / m, the next
    below alpha / (m - 1), and so on, stopping at the first that is not."""
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    passed: set[str] = set()
    for rank, (name, p) in enumerate(ordered):
        if not p < alpha / (len(ordered) - rank):
            break
        passed.add(name)
    return passed


def detectable_gain(answerable: int, compared: int, alpha: float) -> int | None:
    """The fewest more right answers, with none lost, that pass the strictest Holm bar with `compared` arms: the
    smallest k with `mcnemar_p(k, 0) < alpha / compared`. None when that takes more than the `answerable` questions
    there are, so no arm could win."""
    if compared < 1:
        raise ValueError("at least one arm must be compared with the start")
    bar = alpha / compared
    return next((k for k in range(1, answerable + 1) if mcnemar_p(k, 0) < bar), None)
