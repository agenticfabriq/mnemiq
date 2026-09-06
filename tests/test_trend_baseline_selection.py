"""Which recorded run is the bar.

The JSON backend took `rows[-1]` -- the LAST LINE, chosen by position, while the Postgres
backend orders by `run_at DESC`. Two stores that are meant to answer the same question,
disagreeing whenever the file is not in chronological order.

It became load-bearing the moment a superseded row was deliberately left above the row that
supersedes it: a 100.0 sample kept as history above the reproducible 96.0. Reordering that
file, sorting it, or inserting the next accepted level anywhere but the end silently makes
100.0 the level to beat again, and the only symptom is nightlies reddening on one unstable
case -- read as a regression by whoever sees it.
"""

from __future__ import annotations

import json

from mnemiq.eval.trend import last_run


def _write(tmp_path, rows):
    path = tmp_path / "trend.json"
    path.write_text(json.dumps(rows))
    return str(path)


def _row(accuracy: float, run_at: str) -> dict:
    return {
        "source_id": "acme", "run_at": run_at, "accuracy": accuracy,
        "strict_accuracy": accuracy - 0.12, "total": 30, "correct": 21,
        "correct_facts": 3, "wrong": 1, "deferred_wrongly": 0, "error": 0,
    }


def test_the_newest_run_is_the_baseline_whatever_order_the_file_is_in(tmp_path):
    """The mutation this exists for: the superseded row sitting last."""
    path = _write(tmp_path, [
        _row(0.96, "2026-09-06T04:54:40+00:00"),   # newest, and NOT last
        _row(1.00, "2026-09-06T04:16:29+00:00"),   # the sample it supersedes
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96


def test_chronological_order_still_gives_the_same_answer(tmp_path):
    """The ordinary case must not change: `record_run` appends, so files are usually in order."""
    path = _write(tmp_path, [
        _row(1.00, "2026-09-06T04:16:29+00:00"),
        _row(0.96, "2026-09-06T04:54:40+00:00"),
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96


def test_a_legacy_row_with_no_timestamp_never_outranks_a_dated_one(tmp_path):
    """`run_at` was written as `""` before it defaulted to now, and such rows still exist.

    An empty string must not win, in either direction: sorted as text it loses to any ISO
    timestamp, but it must also not be picked just for sitting last.
    """
    path = _write(tmp_path, [
        _row(0.96, "2026-09-06T04:54:40+00:00"),
        _row(0.50, ""),
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96


def test_a_file_of_only_undated_rows_falls_back_to_the_last_one(tmp_path):
    """The pre-`run_at` files this must keep reading, where position is the only ordering there is."""
    path = _write(tmp_path, [_row(0.50, ""), _row(0.96, "")])
    assert last_run(None, "acme", path=path).accuracy == 0.96


def test_another_source_is_not_the_baseline_for_this_one(tmp_path):
    path = _write(tmp_path, [
        {**_row(0.96, "2026-09-06T04:54:40+00:00")},
        {**_row(0.10, "2026-09-06T05:00:00+00:00"), "source_id": "bird"},
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96


def test_stamps_are_compared_as_instants_not_as_text(tmp_path):
    """`Z` and `+00:00` spell the same instant and sort differently as strings.

    Nothing enforces one spelling: `record_run` writes `isoformat()` (`+00:00`), a hand-recorded
    row may use `Z`, and a local offset sorts by its punctuation. Here the LATER instant is
    written `Z` and sorts BEFORE the earlier `+00:00` one as text.
    """
    path = _write(tmp_path, [
        _row(0.60, "2026-09-06T05:00:00+00:00"),
        _row(0.96, "2026-09-06T06:00:00Z"),
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96

    # And an offset that is not UTC: 05:00-07:00 is 12:00Z, later than either above.
    path = _write(tmp_path, [
        _row(0.96, "2026-09-06T05:00:00-07:00"),
        _row(0.60, "2026-09-06T06:00:00Z"),
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96


def test_a_tie_keeps_the_later_row_as_position_did(tmp_path):
    path = _write(tmp_path, [
        _row(0.60, "2026-09-06T05:00:00+00:00"),
        _row(0.96, "2026-09-06T05:00:00+00:00"),
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96


def test_a_row_whose_stamp_cannot_be_parsed_does_not_become_the_baseline(tmp_path):
    path = _write(tmp_path, [
        _row(0.96, "2026-09-06T05:00:00+00:00"),
        _row(0.10, "last Tuesday"),
    ])
    assert last_run(None, "acme", path=path).accuracy == 0.96
