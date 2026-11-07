"""The report: each arm tallied and compared with the start run by run, another system's answers graded beside them by
the same grader, and the values the screen keeps for the grid."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from mnemiq.sweep.arms import grid_arms
from mnemiq.sweep.config import load_config
from mnemiq.sweep.report import grade_control, kept_values, pair, pair_runs, report, tally
from mnemiq.sweep.run import arm_folder
from mnemiq.sweep.stats import mcnemar_p

from .conftest import (
    answers_file, arm_with, cancel_gold_once, read_rows, row, start_of, tiny_dataset, tiny_deployment, tiny_files,
    tuning, two_deployments, write_config, write_run,
)

_KINDS = {"correct": ("correct", {}), "correct_facts": ("correct_facts", {}), "wrong": ("wrong", {}),
          "made_up": ("wrong", {"made_up": True}), "declined": ("deferred_wrongly", {}),
          "failed": ("error", {"error_kind": "answer"}), "errors": ("error", {"error_kind": "endpoint"}),
          "unjudged": ("correct", {"verify_layer": "judge_unavailable"}), "ungraded": (None, {})}


def rows(*, versions=("v1",), mode: str = "instant", **counts: int) -> list[dict]:
    """Rows for ids q0, q1, ... in the order of `_KINDS`, as many of each kind as `counts` names, their versions taken
    in turn from `versions`."""
    kinds = [kind for kind in _KINDS for _ in range(counts.pop(kind, 0))]
    assert not counts, f"no such kind of row: {counts}"
    return [row(f"q{number}", _KINDS[kind][0], mode=mode, version=versions[number % len(versions)], **_KINDS[kind][1])
            for number, kind in enumerate(kinds)]


def pairs(*, start_runs: list[list[int]], arm_runs: list[list[int]]) -> list:
    """Each run of an arm against the start's, from 1 for a right answer and 0 for a wrong one, question by question."""
    def runs(bits: list[list[int]]) -> dict[int, list[dict]]:
        return {number: [row(f"q{i}", "correct" if bit else "wrong") for i, bit in enumerate(run)]
                for number, run in enumerate(bits, start=1)}

    return list(pair_runs(runs(start_runs), runs(arm_runs)).values())


def _changes(label: str) -> dict:
    """`setting=value` read as the config would type the value: `true` a bool, `36` a number, else the text."""
    name, _, value = label.partition("=")
    try:
        return {name: json.loads(value)}
    except ValueError:
        return {name: value}


def sweep_with_rows(tmp_path, *, arms: dict, repeats: int = 1, **sections):
    """A screen already run, written as rows, over 9 answerable tuning cases and 2 to decline.

    The start gets the first R answerable cases right, where R is the largest `other` asked for, and declines the rest.
    Each arm, in each run, is a dict of `only` (right where the start is not), `other` (not right where the start is),
    `more_wrong` and `more_made_up` (answers beyond the start's); a list gives one dict per run. Every not-right answer
    is a decline, so wrong and made-up answers are only those asked for.
    """
    space: dict[str, list] = {}
    for label in arms:
        for name, value in _changes(label).items():
            space.setdefault(name, []).append(value)
    cfg = load_config(write_config(tmp_path, dataset=tiny_dataset(tmp_path, cases=14, answerable=12), space=space,
                                   sweep={"repeats": repeats}, **sections))
    cases = tuning(cfg)
    answerable = [case.id for case in cases if case.answerable]
    unanswerable = [case.id for case in cases if not case.answerable]
    runs = {label: spec if isinstance(spec, list) else [spec] * repeats for label, spec in arms.items()}
    right = max((run["other"] for each in runs.values() for run in each), default=0)
    start = start_of(cfg, "instant")
    started = ([row(case, "correct" if i < right else "deferred_wrongly", version="v-start")
                for i, case in enumerate(answerable)]
               + [row(case, "deferred_correctly", version="v-start") for case in unanswerable])
    for repeat in range(1, repeats + 1):
        write_run(cfg, start, started, repeat)
    for label, each in runs.items():
        arm = arm_with(cfg, **_changes(label))
        for repeat, run in enumerate(each, start=1):
            only, other = run["only"], run["other"]
            wrong, made_up = run.get("more_wrong", 0), run.get("more_made_up", 0)
            assert right + only + wrong <= len(answerable) and made_up <= len(unanswerable)

            def outcome(i: int) -> str:
                if i < other or i >= right + only + wrong:
                    return "deferred_wrongly"
                return "correct" if i < right + only else "wrong"

            answered = [row(case, outcome(i), version=f"v-{arm.profile_id}") for i, case in enumerate(answerable)]
            answered += [row(case, "wrong", made_up=True, version=f"v-{arm.profile_id}") if i < made_up
                         else row(case, "deferred_correctly", version=f"v-{arm.profile_id}")
                         for i, case in enumerate(unanswerable)]
            write_run(cfg, arm, answered, repeat)
    return cfg


def sweep_with_rows_ms(tmp_path, *, rows_ms: list[float]):
    """A screen whose start alone has run, on one answerable case per time to rows given, each answer taking 1.5 s
    longer in all than its time to rows."""
    dataset = tiny_dataset(tmp_path, cases=len(rows_ms), answerable=len(rows_ms)) | {"held_out": 0.0}
    cfg = load_config(write_config(tmp_path, dataset=dataset))
    write_run(cfg, start_of(cfg, "instant"), [row(case.id, "correct", ms=ms + 1500, rows_ms=ms)
                                              for case, ms in zip(tuning(cfg), rows_ms, strict=True)])
    return cfg


def gold_answers(tmp_path) -> Path:
    """Another system's answers to `sweep_with_rows`'s question file: the gold SQL for each case with one, and a decline
    for each without."""
    questions = json.loads(Path(tiny_dataset(tmp_path, cases=14, answerable=12)["questions"]).read_text())
    path = tmp_path / "gold-answers.json"
    path.write_text(json.dumps([{"id": case["id"], "sql": case["gold_sql"]} if case["gold_sql"]
                                else {"id": case["id"], "declined": True} for case in questions]))
    return path


def _record(records: list[dict], label: str) -> dict:
    return next(record for record in records if record["label"] == label)


def test_ungraded_rows_count_against_validity():
    # A row holding no outcome the report can read: no row an arm or a control writes today is one.
    assert tally(rows(correct=7, ungraded=1), "instant", max_errors=0).valid is False


def test_an_answer_that_went_out_while_its_judge_was_down_counts_against_validity():
    t = tally(rows(correct=7, unjudged=1), "instant", max_errors=0)
    assert (t.valid, t.unjudged, t.right) == (False, 1, 7)
    assert tally(rows(correct=7, unjudged=1), "instant", max_errors=1).valid is True
    # a judge that was down fails a fail-closed answer: that is no measure of the arm either
    closed = tally(rows(correct=7) + [row("q7", "error", error_kind="answer", verify_layer="judge_unavailable")],
                   "instant")
    assert (closed.valid, closed.unjudged, closed.failed) == (False, 1, 0)


@pytest.mark.parametrize("kind", ["endpoint", "source"])
def test_an_error_that_left_a_case_unmeasured_counts_against_validity(kind):
    unmeasured = rows(correct=7) + [row("q7", "error", error_kind=kind)]
    t = tally(unmeasured, "instant", max_errors=0)
    assert (t.valid, t.errors, t.failed) == (False, 1, 0)
    assert tally(unmeasured, "instant", max_errors=1).valid is True


def test_the_arms_own_failure_is_measured_and_shown_apart():
    t = tally(rows(correct=6, failed=2), "instant", max_errors=0)
    assert (t.valid, t.right, t.failed, t.errors) == (True, 6, 2, 0)


def test_two_versions_in_one_arm_make_it_invalid():
    assert tally(rows(correct=8, versions=["aaa", "bbb"]), "instant").valid is False


def test_each_deployment_holds_one_version_of_its_own():
    alternate = {f"q{i}": ("lib-a", "lib-b")[i % 2] for i in range(8)}   # versions alternate the same way
    assert tally(rows(correct=8, versions=["aaa", "bbb"]), "instant", cases=alternate).valid is True
    halves = {f"q{i}": ("lib-a", "lib-b")[i // 4] for i in range(8)}
    assert tally(rows(correct=8, versions=["aaa", "bbb"]), "instant", cases=halves).valid is False


def test_an_arm_without_a_row_for_every_case_is_not_valid():
    assert tally(rows(correct=7), "instant", cases={f"q{i}": None for i in range(8)}).valid is False
    assert tally(rows(correct=7), "instant", cases={f"q{i}": None for i in range(7)}).valid is True


def test_made_up_is_counted_apart_from_wrong():
    t = tally(rows(correct=5, wrong=1, made_up=2), "instant")
    assert (t.wrong, t.made_up) == (1, 2)


def test_strict_counts_exact_matches_alone_and_a_decline_is_either_kind():
    declines = [row("q3", "deferred_wrongly"), row("q4", "deferred_correctly")]
    t = tally(rows(correct=2, correct_facts=1) + declines, "instant")
    assert (t.right, t.strict, t.declined) == (3, 2, 2)


def test_a_tally_reads_only_its_own_mode():
    t = tally(rows(correct=3) + rows(wrong=5, mode="thinking"), "instant")
    assert (t.cases, t.right, t.wrong) == (3, 3, 0)


def test_times_are_medians_and_tokens_a_mean_per_question():
    timed = [row("q0", "correct", ms=1000, rows_ms=500, prompt_tokens=100),
             row("q1", "correct", ms=3000, rows_ms=700, prompt_tokens=200),
             row("q2", "deferred_wrongly", ms=2000, prompt_tokens=600)]   # declined: no rows, so no time to them
    t = tally(timed, "instant")
    assert (t.median_ms, t.median_rows_ms, t.prompt_tokens) == (2000, 600, 300)


def test_runs_pair_with_their_own_number():
    assert [p.only for p in pairs(start_runs=[[0, 0, 1], [0, 1, 1]], arm_runs=[[1, 1, 1], [1, 1, 1]])] == [2, 1]


def test_every_answer_that_is_not_right_is_not_right_in_the_pairing():
    not_right = rows(failed=1, declined=1, wrong=1, errors=1, unjudged=1)
    right = [row(r["id"], "correct") for r in not_right]
    assert pair(not_right, right) == (5, 0, mcnemar_p(5, 0))
    assert pair(right, not_right) == (0, 5, mcnemar_p(0, 5))


def test_a_control_is_graded_by_the_same_grader(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, stub_model),
                                   control=[{"name": "current", "answers": answers_file(tmp_path, gold_for=6, declined=2)}]))
    assert sum(r["outcome"] in ("correct", "correct_facts") for r in grade_control(cfg, cfg.control[0])) == 6


def test_a_control_answer_is_run_on_the_source_and_graded(tmp_path):
    answers = answers_file(tmp_path, gold_for=4, wrong_for=2, declined=1)
    cfg = load_config(write_config(tmp_path, control=[{"name": "current", "answers": str(answers)}]))
    graded = grade_control(cfg, cfg.control[0])
    assert [r["outcome"] for r in graded] == ["correct"] * 4 + ["wrong"] * 2 + ["deferred_correctly"]
    assert all(r["profile_version"] is None and r["ms"] is None and r["mode"] is None for r in graded)


def test_a_controls_error_says_whose_it_is(tmp_path, monkeypatch):
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps([{"id": "book-count", "sql": "SELECT COUNT(*) AS books FROM books"},
                                   {"id": "canadian-authors", "sql": "SELECT CAST(name AS INTEGER) AS n FROM authors"},
                                   {"id": "book-prices", "sql": "SELECT title FROM books"}]))
    cfg = load_config(write_config(tmp_path, control=[{"name": "current", "answers": str(answers)}]))
    cancel_gold_once(monkeypatch, "book-count")
    graded = grade_control(cfg, cfg.control[0])
    # the grading's gold query failed; the control's own SQL failed; it answered a question with no answer
    assert [(r["outcome"], r["error_kind"], r["made_up"]) for r in graded] == [
        ("error", "source", False), ("error", "answer", False), ("wrong", None, True)]


def test_a_controls_sql_never_writes_to_the_source(tmp_path):
    d = tiny_files(tmp_path)
    with open(d.env, "a") as fh:
        fh.write("MNEMIQ_WRITE_ENABLED='true'\n")
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps([{"id": "book-count", "sql": "DELETE FROM loans"}]))
    cfg = load_config(write_config(tmp_path, deployment=d, control=[{"name": "current", "answers": str(answers)}]))
    graded = grade_control(cfg, cfg.control[0])
    with sqlite3.connect(d.source) as con:
        assert con.execute("SELECT COUNT(*) FROM loans").fetchone() == (15,)
    assert (graded[0]["outcome"], graded[0]["error_kind"]) == ("error", "answer")


def test_the_table_shows_each_arm_and_control_with_what_it_changed(tmp_path):
    cfg = sweep_with_rows(tmp_path, arms={"writer_rules=none": dict(only=4, other=0)},
                          control=[{"name": "current", "answers": str(gold_answers(tmp_path))}])
    table, records = report(cfg, "screen")
    assert "| Declined | Failed | Errors | Ungraded | Unjudged |" in table
    assert [r["label"] for r in records] == ["start", "writer_rules=none", "current"]
    arm = _record(records, "writer_rules=none")
    assert arm["changes"] == {"writer_rules": "none"} and arm["valid"] is True
    assert (arm["pairs"][0]["only"], arm["pairs"][0]["other"]) == (4, 0)
    # The control answered the held-out cases too; it is graded on the tuning cases, and holds no version.
    control = _record(records, "current")
    assert control["control"] == "current" and control["valid"] is True and control["runs"][0]["versions"] == []
    assert (control["runs"][0]["cases"], control["runs"][0]["right"], control["pairs"][0]["only"]) == (11, 9, 9)
    assert "| control: current | yes | - | 9 |" in table


def test_a_control_is_reported_before_any_arm_has_run(tmp_path):
    answers = answers_file(tmp_path, gold_for=6, declined=2)
    cfg = load_config(write_config(tmp_path, control=[{"name": "current", "answers": str(answers)}]))
    table, records = report(cfg, "screen", repeats=1)
    control = _record(records, "current")
    assert control["valid"] is True and control["runs"][0]["cases"] == len(tuning(cfg)) == 7
    assert control["runs"][0]["right"] == 5 and control["runs"][0]["versions"] == []


def test_a_sweep_over_two_deployments_holds_one_version_for_each(tmp_path, stub_model):
    cfg = load_config(write_config(tmp_path, model=stub_model, dataset=two_deployments(tmp_path, stub_model)))
    reads = {case.id: case.db_id for case in tuning(cfg)}
    assert set(reads.values()) == {"lib-a", "lib-b"}
    write_run(cfg, start_of(cfg, "instant"), [row(case, "correct", version=f"v-{db}") for case, db in reads.items()])
    assert _record(report(cfg, "screen", repeats=1)[1], "start")["valid"] is True


def test_the_grids_arms_are_compared_with_the_screens_start(tmp_path):
    cfg = sweep_with_rows(tmp_path, arms={"writer_rules=none": dict(only=4, other=0), "retrieval_k=36": dict(only=3, other=0)})
    kept = kept_values(cfg, report(cfg, "screen")[1])
    combined = next(arm for arm in grid_arms(cfg, kept) if arm.round == "grid")
    write_run(cfg, combined, read_rows(cfg, arm_with(cfg, writer_rules="none")))
    table, records = report(cfg, "grid")
    assert "| Arm | Valid | Right |" in table and "| Run 1 only/other | Run 1 p |" in table
    assert "| 4/0 | 0.125 |" in table
    assert [r["label"] for r in records] == ["start", "retrieval_k=36 + writer_rules=none"]
    assert _record(records, "start")["round"] == "screen"
    assert _record(records, "retrieval_k=36 + writer_rules=none")["pairs"][0]["only"] == 4


def test_the_screen_keeps_a_value_that_helps_and_drops_one_that_harms(tmp_path):
    cfg = sweep_with_rows(tmp_path, arms={"writer_rules=none": dict(only=4, other=0),
                                          "guided_sql=true": dict(only=4, other=0, more_wrong=1)})
    table, records = report(cfg, "screen")
    assert kept_values(cfg, records)[("m", "instant")] == {"writer_rules": ["none"]}
    # the screen's table shows the p it decides by, one-sided, and marks what it keeps
    assert "| Arm | Valid | Kept | Right |" in table and "| Run 1 only/other | Run 1 p (one-sided) |" in table
    assert "| writer_rules=none | yes | yes |" in table and "| guided_sql=true | yes | no |" in table
    assert "| 4/0 | 0.0625 |" in table and "| start | yes | - |" in table


@pytest.mark.parametrize("only, other, kept", [
    (3, 0, True),    # one-sided p 0.125
    (2, 0, False),   # 0.25, not below the bar of 0.25
    (5, 2, True),    # 0.227: its two-sided p, 0.453, would drop it
    (4, 4, False),   # the arm must win more than it loses
])
def test_the_screens_bar_is_a_one_sided_p_strictly_below_screen_alpha(tmp_path, only, other, kept):
    cfg = sweep_with_rows(tmp_path, arms={"retrieval_k=36": dict(only=only, other=other)})
    table, records = report(cfg, "screen")
    assert kept_values(cfg, records)[("m", "instant")] == ({"retrieval_k": [36]} if kept else {})
    assert f"| retrieval_k=36 | yes | {'yes' if kept else 'no'} |" in table


def test_a_value_that_wins_no_more_than_it_loses_is_dropped_under_any_bar(tmp_path):
    # Four won and four lost: by chance at least four wins of eight is 0.637, under a bar of 0.9.
    cfg = sweep_with_rows(tmp_path, arms={"retrieval_k=36": dict(only=4, other=4)}, search={"screen_alpha": 0.9})
    table, records = report(cfg, "screen")
    assert kept_values(cfg, records)[("m", "instant")] == {} and "| retrieval_k=36 | yes | no |" in table


@pytest.mark.parametrize("propose, kept", [({}, False), ({"max_more_made_up": 1}, True)])
def test_the_screen_holds_made_up_answers_to_the_propose_bound(tmp_path, propose, kept):
    cfg = sweep_with_rows(tmp_path, arms={"writer_rules=none": dict(only=4, other=0, more_made_up=1)}, propose=propose)
    assert kept_values(cfg, report(cfg, "screen")[1])[("m", "instant")] == ({"writer_rules": ["none"]} if kept else {})


def test_the_screen_keeps_a_value_only_when_every_run_passes(tmp_path):
    cfg = sweep_with_rows(tmp_path, repeats=2, arms={"writer_rules=none": [dict(only=4, other=0), dict(only=1, other=1)],
                                                     "retrieval_k=36": [dict(only=4, other=0), dict(only=3, other=0)]})
    table, records = report(cfg, "screen")
    assert "| Run 2 only/other | Run 2 p (one-sided) |" in table and "| writer_rules=none | yes | no | 5 / 1 |" in table
    assert kept_values(cfg, records)[("m", "instant")] == {"retrieval_k": [36]}


def test_an_arm_short_of_a_run_is_not_valid(tmp_path):
    cfg = sweep_with_rows(tmp_path, repeats=2, arms={"writer_rules=none": dict(only=4, other=0)})
    arm = arm_with(cfg, writer_rules="none")
    shutil.rmtree(arm_folder(cfg, arm, 2))
    records = report(cfg, "screen")[1]
    assert _record(records, "writer_rules=none")["valid"] is False and kept_values(cfg, records)[("m", "instant")] == {}


def test_a_run_whose_pinned_files_moved_is_not_valid(tmp_path):
    cfg = sweep_with_rows(tmp_path, arms={"writer_rules=none": dict(only=4, other=0)})
    (arm_folder(cfg, arm_with(cfg, writer_rules="none"), 1) / "invalid.json").write_text('{"changed": ["store"]}')
    records = report(cfg, "screen")[1]
    assert _record(records, "writer_rules=none")["valid"] is False and kept_values(cfg, records)[("m", "instant")] == {}


def test_no_value_is_kept_against_a_start_that_is_not_valid(tmp_path):
    cfg = sweep_with_rows(tmp_path, arms={"writer_rules=none": dict(only=4, other=0)})
    path = arm_folder(cfg, start_of(cfg, "instant"), 1) / "rows.jsonl"
    started = [json.loads(line) for line in path.read_text().splitlines()]
    started[0] |= {"outcome": "error", "error_kind": "endpoint"}
    path.write_text("".join(json.dumps(r) + "\n" for r in started))
    records = report(cfg, "screen")[1]
    assert _record(records, "start")["valid"] is False and kept_values(cfg, records)[("m", "instant")] == {}


def test_the_report_leaves_a_row_still_being_written_as_it_is(tmp_path):
    cfg = sweep_with_rows(tmp_path, arms={"writer_rules=none": dict(only=4, other=0)})
    path = arm_folder(cfg, arm_with(cfg, writer_rules="none"), 1) / "rows.jsonl"
    with open(path, "a") as fh:
        fh.write('{"id": "after-19')
    before = path.read_text()
    records = report(cfg, "screen")[1]
    assert path.read_text() == before and _record(records, "writer_rules=none")["valid"] is True


def test_a_report_under_auto_repeats_needs_the_count_the_first_round_decided(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.sweep.repeats == "auto"
    with pytest.raises(ValueError, match="repeats"):
        report(cfg, "screen")
    assert report(cfg, "screen", repeats=1) == ("", [])


def test_a_report_of_the_timing_round_is_refused(tmp_path):
    cfg = load_config(write_config(tmp_path, sweep={"repeats": 1}))
    with pytest.raises(ValueError, match="timing"):
        report(cfg, "timing")


def test_the_table_has_time_to_rows(tmp_path):
    table, _ = report(sweep_with_rows_ms(tmp_path, rows_ms=[2000, 4000, 6000]), "screen", repeats=1)
    assert "| Median s to rows |" in table and "| 4.0 |" in table
