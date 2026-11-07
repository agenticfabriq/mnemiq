"""The split: the questions a sweep tunes on and the ones it keeps back, fixed by a seed."""

from __future__ import annotations

from mnemiq.sweep.split import split

from .conftest import answerable, cases, config_with


def test_the_split_floors_the_held_out_share():
    parts = split(config_with(held_out=0.3), cases(answerable=17, unanswerable=9))
    assert (answerable(parts["tuning"]), answerable(parts["held_out"])) == (12, 5)


def test_check_mode_tunes_on_every_case():
    every = cases(answerable=17, unanswerable=9)
    parts = split(config_with(strategy="check"), every)
    assert parts["tuning"] == every and parts["held_out"] == []


def test_the_seed_fixes_the_split_and_the_files_order_does_not():
    every = cases(answerable=17, unanswerable=9)

    def held(seed, some):
        return {case.id for case in split(config_with(split_seed=seed), some)["held_out"]}

    assert held(7, every) == held(7, list(reversed(every)))
    assert held(7, every) != held(8, every)


def test_the_share_is_read_as_written():
    # 0.29 * 100 is 28.999999999999996 in floating point, which floors to 28
    parts = split(config_with(held_out=0.29), cases(answerable=100, unanswerable=0))
    assert len(parts["held_out"]) == 29
