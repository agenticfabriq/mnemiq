"""What a sweep's rows say: each arm tallied and compared with the start question by question, another system's answers
graded beside them, a table for each model and mode, and the values the screen keeps for the grid.

Each row counts as exactly one of these:

- right (`correct` or `correct_facts`; `correct` alone is strict), wrong, made up (an answer to a question whose case
  says it has none), or declined;
- failed: the arm's own failure (`error_kind` "answer"), such as SQL the source rejected at every repair. The case was
  measured, and the arm did not get it right;
- an error: a model endpoint or the source failed us (`error_kind` "endpoint" or "source"), so the case was not
  measured;
- unjudged: the answer went out while its judge could not be reached (`verify_layer` "judge_unavailable"), so it is no
  measure of the arm as it was set, whatever the grader made of it;
- ungraded: the row holds no outcome this report can read. `run_case` gives every row one of its outcomes, a grading
  that could not be done included (an error, which the row says is the source's), so no row an arm or a control writes
  today is ungraded.

Errors, unjudged and ungraded rows are the cases an arm did not measure, and an arm with more of them than
`max_errors` is not valid. Paired with the start, a row is right or it is not: failed, declined, wrong, an error and
unjudged are all not right.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import IdentityContext, Trace
from mnemiq.eval.harness import CaseResult, Outcome, run_case
from mnemiq.runtime import source_adapter
from mnemiq.sweep.arms import ROUNDS, arm_from_profile
from mnemiq.sweep.config import Control, SweepConfig, control_answers, deployment_settings, load_questions
from mnemiq.sweep.run import INVALID, PROFILE, ROWS, _identity, _rows_so_far
from mnemiq.sweep.split import split
from mnemiq.sweep.stats import mcnemar_p, mcnemar_p_one_sided

_RIGHT = frozenset({Outcome.CORRECT, Outcome.CORRECT_FACTS})
_DECLINED = frozenset({Outcome.DEFERRED_CORRECTLY, Outcome.DEFERRED_WRONGLY})
_READ = _RIGHT | _DECLINED | {Outcome.WRONG, Outcome.ERROR}
_UNMEASURED = ("errors", "unjudged", "ungraded")
# The part of the questions each round runs on, when it is not the tuning part.
_ROUND_PART = {"confirm": "held_out"}
_COLUMNS = ("Right", "Strict", "Wrong", "Made up", "Declined", "Failed", "Errors", "Ungraded", "Unjudged", "Median s",
            "Median s to rows", "Prompt tokens")
_COUNTED = ("right", "strict", "wrong", "made_up", "declined", "failed", "errors", "ungraded", "unjudged")


@dataclass(frozen=True)
class Tally:
    """One run of an arm, or a control's answers, in one mode. Times are medians over the rows that have one, and
    prompt tokens a mean over the rows that count them."""

    valid: bool
    cases: int
    right: int
    strict: int
    wrong: int
    made_up: int
    declined: int
    failed: int
    errors: int
    ungraded: int
    unjudged: int
    median_ms: float | None
    median_rows_ms: float | None
    prompt_tokens: float | None  # per question
    versions: list[str]


class Pair(NamedTuple):
    """One run of an arm against the same run of the start: the questions the arm alone got right (`only`), those the
    start alone did (`other`), and the exact two-sided McNemar p."""

    only: int
    other: int
    p: float


def _kind(row: Mapping) -> str:
    """What one row counts as, in the module's terms."""
    outcome = row.get("outcome")
    if outcome not in _READ:
        return "ungraded"
    if outcome == Outcome.ERROR and row.get("error_kind") != "answer":
        return "errors"  # the endpoint's, the source's, or a row that does not say whose
    if row.get("verify_layer") == "judge_unavailable":
        return "unjudged"
    if outcome == Outcome.ERROR:
        return "failed"
    if outcome in _RIGHT:
        return "right"
    if outcome == Outcome.WRONG:
        return "made_up" if row.get("made_up") else "wrong"
    return "declined"  # the outcomes `_READ` leaves are the two declines


def _right(row: Mapping) -> bool:
    return _kind(row) == "right"


def _median(values: Iterable[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return statistics.median(known) if known else None


def _mean(values: Iterable[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return statistics.fmean(known) if known else None


def tally(rows: Iterable[Mapping], mode: str, *, cases: Mapping[str, str | None] | None = None, max_errors: int = 0,
          invalid: bool = False, versioned: bool = True) -> Tally:
    """The rows of one run in `mode`, counted.

    `cases` maps each case the run owes to the deployment it reads (None with one env file); by default it is the rows'
    own cases, read as one deployment. The run is valid when it holds one row for each case and no other, the cases it
    did not measure are at most `max_errors`, it is not `invalid` (a pinned file moved), and, when `versioned`, the rows
    of each deployment hold one profile version: a profile's version holds deployment settings, so a sweep over several
    deployments may stamp one version each. A control's rows hold no version.
    """
    mine = [row for row in rows if row.get("mode") == mode]
    owed = dict(cases) if cases is not None else {row.get("id"): None for row in mine}
    ids = [row.get("id") for row in mine]
    kinds = Counter(_kind(row) for row in mine)
    by_deployment: dict[str | None, set] = defaultdict(set)
    for row in mine:
        by_deployment[owed.get(row.get("id"))].add(row.get("profile_version"))
    one_version = all(len(held) == 1 and None not in held for held in by_deployment.values())
    complete = bool(owed) and len(ids) == len(set(ids)) and set(ids) == set(owed)
    unmeasured = sum(kinds[kind] for kind in _UNMEASURED)
    return Tally(
        valid=complete and not invalid and (one_version or not versioned) and unmeasured <= max_errors,
        cases=len(owed),
        right=kinds["right"],
        strict=sum(_right(row) and row.get("outcome") == Outcome.CORRECT for row in mine),
        wrong=kinds["wrong"],
        made_up=kinds["made_up"],
        declined=kinds["declined"],
        failed=kinds["failed"],
        errors=kinds["errors"],
        ungraded=kinds["ungraded"],
        unjudged=kinds["unjudged"],
        median_ms=_median(row.get("ms") for row in mine),
        median_rows_ms=_median(row.get("rows_ms") for row in mine),
        prompt_tokens=_mean(row.get("prompt_tokens") for row in mine),
        versions=sorted({row["profile_version"] for row in mine if row.get("profile_version") is not None}),
    )


def pair(start_rows: Iterable[Mapping], arm_rows: Iterable[Mapping]) -> Pair:
    """One run of an arm against one run of the start, over the cases and modes both answered."""
    start = {(row.get("id"), row.get("mode")): _right(row) for row in start_rows}
    only = other = 0
    for row in arm_rows:
        key = (row.get("id"), row.get("mode"))
        if key in start:
            only += _right(row) and not start[key]
            other += start[key] and not _right(row)
    return Pair(only, other, mcnemar_p(only, other))


def pair_runs(start_runs: Mapping[int, Sequence[Mapping]], arm_runs: Mapping[int, Sequence[Mapping]]) -> dict[int, Pair]:
    """Each run of an arm against the start's run of the same number, for each number both have: run k of an arm ran
    beside run k of the start, under the same load on the server."""
    return {number: pair(start_runs[number], arm_runs[number]) for number in sorted(arm_runs) if number in start_runs}


def _answering(answer: dict, identity: IdentityContext):
    """An engine that gives `answer` to whatever it is asked: the control's SQL, as an answer's trace carries what ran,
    or a decline."""
    def engine(question: str) -> AgentAnswer:
        if answer.get("declined"):
            return AgentAnswer(answer="(the control declined)", deferred=True)
        trace = Trace(question=question, plan_sql=answer["sql"], target_sql=answer["sql"], result_shape="table",
                      timing={}, enrichment_version="control", identity=identity)
        return AgentAnswer(answer="(the control's SQL)", trace=trace)

    return engine


def _control_error_kind(result: CaseResult) -> str | None:
    """Who failed a control's row that is an error. The gold ran and the control's SQL did not: the control's own
    failure, as an arm's SQL the source rejects is the arm's. Otherwise the grading failed, the gold query or a reading of
    the two results, and that is the source's."""
    if result.outcome != Outcome.ERROR:
        return None
    return "answer" if result.gold_row_count is not None and not result.executed else "source"


def grade_control(cfg: SweepConfig, control: Control) -> list[dict]:
    """A control's answers, read as `load_config` checked them (`control_answers`), each run on its deployment's adapter
    and graded by `run_case`, as rows like an arm's with no mode, no version, no time and no tokens. A decline is graded
    as a decline. The adapter is read-only whatever the deployment allows, since nothing checks a control's SQL before
    it runs."""
    cases = {case.id: case for case in load_questions(cfg)}
    answers = control_answers(control, cases)
    identity = _identity(cfg)
    adapters: dict[str | None, object] = {}
    rows = []
    try:
        for answer in answers:
            case = cases[answer["id"]]
            db_id = case.db_id if cfg.dataset.deployments is not None else None
            if db_id not in adapters:
                adapters[db_id] = source_adapter(deployment_settings(cfg, db_id), read_only=True)
            result = run_case(case, _answering(answer, identity), adapters[db_id])
            rows.append({
                "id": case.id, "mode": None, "outcome": str(result.outcome),
                "made_up": not case.answerable and result.outcome == Outcome.WRONG, "sql": result.sql,
                "ms": None, "rows_ms": None, "calls": None, "prompt_tokens": None, "completion_tokens": None,
                "profile_version": None, "engine_build": None, "error_kind": _control_error_kind(result),
                "model_error": None, "verify_layer": None,
            })
    finally:
        for adapter in adapters.values():
            close = getattr(adapter, "close", None)
            if close is not None:
                close()
    return rows


@dataclass(frozen=True)
class _Run:
    rows: list[dict]
    invalid: bool


def _run_numbers(folder: Path) -> list[int]:
    found = (path.name.removeprefix("run-") for path in folder.glob("run-*") if path.is_dir())
    return sorted(int(number) for number in found if number.isdigit())


def _arm_folders(cfg: SweepConfig, model: str, round: str, mode: str) -> list[tuple[str, tuple[str, ...], Path]]:
    """Each arm `run_arm` ran for `model` in `round` with `mode` among its modes, as its profile id, its modes and its
    folder (`arm_folder` less the run)."""
    base = Path(cfg.sweep.out) / model / round
    if not base.is_dir():
        return []
    found = []
    for modes_folder in sorted(path for path in base.iterdir() if path.is_dir()):
        modes = tuple(modes_folder.name.split("+"))
        if mode in modes:
            found += [(arm.name, modes, arm) for arm in sorted(path for path in modes_folder.iterdir() if path.is_dir())]
    return found


def _runs(folder: Path, mode: str, count: int) -> dict[int, _Run]:
    """Runs 1 to `count` of the arm in `folder`, those that have a folder, with their rows in `mode`."""
    return {number: _Run([row for row in _rows_so_far(folder / f"run-{number}" / ROWS) if row.get("mode") == mode],
                         (folder / f"run-{number}" / INVALID).exists())
            for number in range(1, count + 1) if (folder / f"run-{number}").is_dir()}


def _start_round(cfg: SweepConfig, round: str) -> str:
    """The round whose start the arms of `round` are compared with. A grid that follows a screen has no start of its
    own, and is compared with the screen's; a grid run with no screen before it runs its own start."""
    return "screen" if round == "grid" and cfg.search.strategy == "screen-then-grid" else round


def _owed(cfg: SweepConfig, round: str) -> dict[str, str | None]:
    """The cases each arm of `round` owes a row for, each to the deployment it reads."""
    part = split(cfg, load_questions(cfg))[_ROUND_PART.get(round, "tuning")]
    return {case.id: case.db_id if cfg.dataset.deployments is not None else None for case in part}


def _against(pairs: dict[int, Pair], tallies: Mapping[int, Tally], start: Mapping[int, Tally]) -> list[dict]:
    """Each run's pair with the start, and how many more wrong and made-up answers it gave than that run of the
    start."""
    return [{"run": number, "only": found.only, "other": found.other, "p": found.p,
             "more_wrong": tallies[number].wrong - start[number].wrong,
             "more_made_up": tallies[number].made_up - start[number].made_up}
            for number, found in pairs.items()]


def _table_records(cfg: SweepConfig, model: str, mode: str, round: str, owed: Mapping[str, str | None],
                   controls: list[tuple[str, list[dict]]], repeats: int) -> list[dict]:
    start_round = _start_round(cfg, round)
    start = next((folder for profile_id, _, folder in _arm_folders(cfg, model, start_round, mode)
                  if profile_id == "start"), None)
    arms = [found for found in _arm_folders(cfg, model, round, mode) if found[0] != "start"]

    def counted(runs: Mapping[int, _Run]) -> dict[int, Tally]:
        return {number: tally(runs[number].rows if number in runs else [], mode, cases=owed,
                              max_errors=cfg.sweep.max_errors, invalid=number in runs and runs[number].invalid)
                for number in range(1, repeats + 1)}

    def record(arm: str | None, label: str, control: str | None, changes: dict, runs: Mapping[int | None, Tally],
               against: list[dict]) -> dict:
        return {"model": model, "mode": mode, "round": start_round if arm == "start" else round, "arm": arm,
                "label": label, "control": control, "changes": changes,
                "valid": all(each.valid for each in runs.values()),
                "runs": [{"run": number, **asdict(each)} for number, each in runs.items()], "pairs": against}

    records = []
    start_runs = _runs(start, mode, repeats) if start is not None else {}
    start_tallies = counted(start_runs)
    if start is not None:
        records.append(record("start", "start", None, {}, start_tallies, []))
    start_rows = {number: run.rows for number, run in start_runs.items()}
    found = []
    for profile_id, modes, folder in arms:
        profile = next((folder / f"run-{number}" / PROFILE for number in _run_numbers(folder)
                        if (folder / f"run-{number}" / PROFILE).exists()), None)
        arm = arm_from_profile(cfg, model, round, modes, profile_id, profile) if profile is not None else None
        runs = _runs(folder, mode, repeats)
        tallies = counted(runs)
        pairs = pair_runs(start_rows, {number: run.rows for number, run in runs.items()})
        found.append(record(profile_id, arm.label if arm is not None else profile_id, None,
                            dict(arm.changes) if arm is not None else {}, tallies,
                            _against(pairs, tallies, start_tallies)))
    records += sorted(found, key=lambda each: each["label"])
    for name, rows in controls:
        # One set of answers, read against each run of the start.
        stamped = [row | {"mode": mode} for row in rows if row["id"] in owed]
        once = tally(stamped, mode, cases=owed, max_errors=cfg.sweep.max_errors, versioned=False)
        pairs = {number: pair(start_rows[number], stamped) for number in sorted(start_rows)}
        records.append(record(None, name, name, {}, {None: once},
                              _against(pairs, {number: once for number in pairs}, start_tallies)))
    return records


def _seconds(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}"


def _cells(record: dict, repeats: int, *, kept: str | None, one_sided: bool) -> list[str]:
    """A record's row of the table. `kept` is its Kept cell, None in a round with no such column; `one_sided` shows each
    run's one-sided p in place of the two-sided one."""
    def each(show) -> str:
        return " / ".join(show(run) for run in record["runs"])

    label = f"control: {record['label']}" if record["control"] is not None else record["label"]
    cells = [label.replace("|", "\\|"), "yes" if record["valid"] else "no", *([kept] if kept is not None else [])]
    cells += [each(lambda run, name=name: str(run[name])) for name in _COUNTED]
    cells += [each(lambda run: _seconds(run["median_ms"])), each(lambda run: _seconds(run["median_rows_ms"])),
              each(lambda run: "-" if run["prompt_tokens"] is None else f"{run['prompt_tokens']:.0f}")]
    pairs = {found["run"]: found for found in record["pairs"]}
    for number in range(1, repeats + 1):
        found = pairs.get(number)
        if found is None:
            cells += ["-", "-"]
            continue
        p = mcnemar_p_one_sided(found["only"], found["other"]) if one_sided else found["p"]
        cells += [f"{found['only']}/{found['other']}", f"{p:.3g}"]
    return cells


def _line(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(cfg: SweepConfig, model: str, mode: str, round: str, records: list[dict], repeats: int) -> str:
    """One model and mode's table. The screen's shows the one-sided p it decides by, and marks each arm it keeps by
    the rule `kept_values` keeps by."""
    screen = round == "screen"
    start = next((record for record in records if record["arm"] == "start"), None)
    p = "p (one-sided)" if screen else "p"
    columns = ["Arm", "Valid", *(["Kept"] if screen else []), *_COLUMNS,
               *(f"Run {number} {what}" for number in range(1, repeats + 1) for what in ("only/other", p))]
    lines = [f"## {model}, {mode}: {round}", "", _line(columns), "|" + "|".join("---" for _ in columns) + "|"]
    for record in records:
        kept = None
        if screen:
            arm = record["control"] is None and record["arm"] != "start"
            kept = ("yes" if _kept(cfg, record, start) else "no") if arm else "-"
        lines.append(_line(_cells(record, repeats, kept=kept, one_sided=screen)))
    return "\n".join(lines)


def report(cfg: SweepConfig, round: str, *, repeats: int | None = None) -> tuple[str, list[dict]]:
    """A markdown table for each model and mode of `round`, and its rows as records.

    A table holds the start, each arm the round ran, then each control. Its columns are Valid, Right, Strict, Wrong,
    Made up, Declined, Failed, Errors, Ungraded, Unjudged, Median s, Median s to rows and Prompt tokens (per question),
    each cell giving every run's value in turn, and, against the start, only/other and p for each run. The screen's
    table also marks the arms whose value it keeps (Kept), and its p is the one-sided p it decides by; every other
    round's is the two-sided p. A grid that follows a screen is compared with the screen's start. The timing round is
    refused: its arms answer only the cases timed.

    Each arm's label and changes are read back from the profile file its run wrote (`arm_from_profile`), and an arm
    folder this config would not have written is refused.

    `repeats` is how many runs each arm owes, and an arm short of one is not valid. By default it is the config's
    `repeats` when that is a number. Under `auto` it must be passed, as the first round decided it: the folders alone
    cannot tell an arm's missing run from a second run no arm has had yet.

    A record holds the model, mode, round, `arm` (its profile id; None for a control), `label`, `control` (its name, or
    None), `changes` (each setting the arm changes from the start, to its value), `valid` (in every run), `runs` (each
    run's tally, with its number; a control's one tally has none) and `pairs` (each run against the start's run of the
    same number: `only`, `other`, `p`, `more_wrong` and `more_made_up`).
    """
    if round not in ROUNDS:
        raise ValueError(f"{round!r} is not a round; the rounds are {', '.join(ROUNDS)}")
    if round == "timing":
        raise ValueError("the timing round is not reported: its arms answer only the cases timed, and this report does "
                         "not know which those are")
    if repeats is None:
        if not isinstance(cfg.sweep.repeats, int):
            raise ValueError('[sweep] repeats is "auto": pass repeats=, the number of runs the first round decided '
                             "each arm owes")
        repeats = cfg.sweep.repeats
    if repeats < 1:
        raise ValueError("an arm owes at least one run")
    owed = _owed(cfg, round)
    controls = [(control.name, grade_control(cfg, control)) for control in cfg.control]
    tables, records = [], []
    for model in cfg.models:
        for mode in cfg.modes:
            mine = _table_records(cfg, model, mode, round, owed, controls, repeats)
            if mine:
                tables.append(_table(cfg, model, mode, round, mine, repeats))
                records += mine
    return "\n\n".join(tables) + ("\n" if tables else ""), records


def _kept(cfg: SweepConfig, record: dict, start: dict | None) -> bool:
    """Whether the screen keeps a one-change arm's value: the arm and its start are valid, and in every run the arm
    wins more of the discordant questions than it loses, with a one-sided p below `screen_alpha`, and adds no more wrong
    or made-up answers than `[propose]` allows."""
    if start is None or not start["valid"] or not record["valid"] or len(record["changes"]) != 1:
        return False
    pairs = record["pairs"]
    return len(pairs) == len(record["runs"]) and all(
        found["only"] > found["other"]
        and mcnemar_p_one_sided(found["only"], found["other"]) < cfg.search.screen_alpha
        and found["more_wrong"] <= cfg.propose.max_more_wrong
        and found["more_made_up"] <= cfg.propose.max_more_made_up
        for found in pairs)


def kept_values(cfg: SweepConfig, records: Iterable[dict]) -> dict[tuple[str, str], dict[str, list]]:
    """For each model and mode the screen ran, each setting to the values its screen kept, in `grid_arms`'s form."""
    records = list(records)
    starts = {(record["model"], record["mode"]): record for record in records
              if record["round"] == "screen" and record["arm"] == "start"}
    kept: dict[tuple[str, str], dict[str, list]] = {}
    for record in records:
        if record["round"] != "screen" or record["control"] is not None:
            continue
        values = kept.setdefault((record["model"], record["mode"]), {})
        if record["arm"] != "start" and _kept(cfg, record, starts.get((record["model"], record["mode"]))):
            ((name, value),) = record["changes"].items()
            values.setdefault(name, []).append(value)
    return kept
