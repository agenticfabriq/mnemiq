"""The proposal: from a sweep's rows, the profile each model should run, and the evidence behind it, for a person to
promote.

The rule is the config's `[propose]`, fixed before any row was written. An arm is a candidate in its mode when it is
valid in every run and, in every run, gets more right than the start where the two differ, with an exact McNemar p that
passes Holm's correction at `alpha` over every distinct arm compared with that start on the cases to tune on, screen and
grid together, and adds no more wrong or made-up answers than `[propose]` allows. A model and mode whose search the
sweep's record says stopped for want of room has no candidate, whatever the arms that ran beside its start's first run
scored. The candidates with the most right answers on average are timed, each once and alone on the first cases to
tune on, beside the start. A tie goes to the lower median time of a reading that was taken on a server that publishes
its load, then to fewer prompt tokens.

A profile holds every mode, so each mode's pick is assembled into one profile (`assembled_arm`), a profile-wide setting
the picks disagree on keeping the start's value. That one arm is confirmed against the start on the held-out cases, in
every mode the sweep runs and once per repeat, run k beside run k. It passes when, in every mode and run, it is valid,
gets at least the start's right answers, and adds no more wrong or made-up answers than the bounds. When it fails, the
start is proposed: no second candidate is tried, since that would make the held-out cases part of the search.

Check mode compares the start with the engine's defaults for harm alone. It proposes the start when, in every run, the
start's wrong and made-up answers exceed the defaults' by no more than the bounds, the defaults otherwise, and nothing
unless both arms are valid in every run.

Each of these spans the modes whose start the sweep ran, as the first round's plan recorded them (`starts` in
`sweep.json`), never asked of the engine again. A mode the engine refused the start in ran nothing, so the profile holds
no section for it: there it keeps the engine's own per-mode settings, and runs under the profile's whole-profile ones.
The evidence and its reason name that mode, and the whole-profile changes from the start never measured there. So they
do for each of the engine's modes the sweep did not search, which the profile holds no section for either. A search
whose start is not valid proposes nothing where the rule would fall back to it, as check mode does.

What is written is the own file of an arm that ran, renamed to `[propose] profile` (the model's name by default): the
confirmed arm's, or the start's or the defaults'. Its version is recomputed by the engine's own loader and version
function, and a proposal whose version is not the one the arm's rows were stamped with is refused. The version hashes
what a profile resolves to, never its name, so the renaming cannot move it. Where one arm ran in every mode (the
confirmed arm, the confirm round's start, any arm of a sweep of one mode), that check is made on the file as written.
Where each mode ran an arm of its own (the start or check mode's arms in a sweep of several modes), the file is put
together from those arms' files, and each is checked against its own mode's rows; the whole file's version then cannot
be one a run stamped, since it hashes every mode, and the evidence says the profile was measured mode by mode.

A proposal's two files, the profile and its evidence, are written together after every check has passed. A refusal
for what was asked, or for a sweep that has not moved, leaves an earlier proposal as it was. A refusal that shows an
earlier proposal no longer holds (an arm's file that cannot be read back as that arm, or whose version is not the one
its rows were stamped with, a split that is not the one drawn, a record that is not finished), or a failure while the
files are written, removes it, so none is left looking current.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import tempfile
import threading
import time
import tomllib
from collections import defaultdict, deque
from collections.abc import Collection, Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from mnemiq import redact
from mnemiq.agent.modes import MODES
from mnemiq.config import without_credentials
from mnemiq.profiles import load_profiles, profile_version
from mnemiq.runtime import source_adapter
from mnemiq.sweep.arms import Arm, _endpoint, _mode_of, _show, _toml, arm_of, assembled_arm, baselines, check_arms
from mnemiq.sweep.config import (
    ENDPOINT_FACTS, PROPOSAL, SWEEP, SweepConfig, SweepConfigError, fingerprint, load_questions, scrub_credentials,
    unset_keys,
)
from mnemiq.sweep.plan import _ROUNDS_OF, LOAD_WAIT_EVERY_S, LOAD_WAIT_S, TIMING_CASES, unpinned_sources
from mnemiq.sweep.report import ArmFileError, decided_repeats, report
from mnemiq.sweep.run import (
    OUT_LOCK, PROFILE, ROWS, _first_round, _replace, _rows_so_far, _say, _sole, _write_json, arm_folder, engine_build,
)
from mnemiq.sweep.split import PARTS, saved_split
from mnemiq.sweep.stats import detectable_gain, holm

PROFILES = "profiles.toml"
EVIDENCE = "evidence.json"
# What the timing round writes in each timed run's folder: the worker's check just before each case, and the sweep's own
# samples of the machine's load while the run lasts.
CHECKS = "checks.jsonl"
LOADS = "load.jsonl"
LOAD_EVERY_S = 30.0
_METRICS_TIMEOUT_S = 2.0
_METRICS_BYTES = 4 * 1024 * 1024
_RUNNING = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE)


class _Stale(SweepConfigError):
    """A refusal that shows an earlier proposal no longer holds, so its files are removed."""


# --- the timing round's checks ----------------------------------------------------------------------------------------


def server_running(base_url: str) -> int | None:
    """The requests a model server is running, as vLLM's `vllm:num_requests_running` reads on its `/metrics` page, at
    the server's root beside `/v1`; None where it publishes no such count, as a hosted endpoint does, or where the page
    cannot be read within a couple of seconds. No key is sent. The reply is read against a deadline and a size cap, so
    a proxy dripping a catch-all page cannot hold the round."""
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    started = time.monotonic()
    try:
        with httpx.stream("GET", f"{root}/metrics", timeout=_METRICS_TIMEOUT_S, follow_redirects=False) as response:
            if response.status_code != 200:
                return None
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > _METRICS_BYTES or time.monotonic() - started > _METRICS_TIMEOUT_S:
                    return None
                chunks.append(chunk)
    except httpx.HTTPError:
        return None
    found = _RUNNING.findall(b"".join(chunks).decode("utf-8", errors="replace"))
    try:
        return round(sum(float(value) for value in found)) if found else None
    except (ValueError, OverflowError):  # a count that is not a number, or not a finite one
        return None


def machine_load() -> tuple[float, int]:
    """The sweep machine's 1-minute load average, and its core count."""
    return os.getloadavg()[0], os.cpu_count() or 1


def quiet_machine(*, load=None, clock=None, sleep=None) -> bool:
    """Wait up to `LOAD_WAIT_S` for the sweep machine's 1-minute load (`machine_load`, or `load`) to fall under its
    core count, reading it every `LOAD_WAIT_EVERY_S`, and say so. The load average lags the work that raised it, so
    without the wait the parallel rounds just before would cost the first timed reading. True once the load is under
    the cores; False when it never fell, and the timing round goes on, its checks deciding what is taken. `clock` and
    `sleep` default to `time.monotonic` and `time.sleep`."""
    read, clock, sleep = load or machine_load, clock or time.monotonic, sleep or time.sleep
    limit, every = LOAD_WAIT_S, LOAD_WAIT_EVERY_S
    current, cores = read()
    if current < cores:
        return True
    _say(f"The machine's 1-minute load is {current:.1f} on {cores} cores: waiting up to {limit:g} s for it to fall "
         "before the timing round.")
    started = clock()
    while clock() - started < limit:
        sleep(every)
        current, cores = read()
        if current < cores:
            _say(f"The load is {current:.1f} after {clock() - started:.0f} s of waiting; the timing round starts.")
            return True
    _say(f"The load is still {current:.1f} on {cores} cores after {limit:g} s: timing anyway, and a reading the load "
         "fails is not taken.")
    return False


@dataclass(frozen=True)
class TimingCheck:
    """What the worker running a timed arm calls just before each case: how many requests the model server at
    `base_url` is running (`server_running`) and the sweep machine's load (`machine_load`), appended to `path` as one
    line. It is sent to the worker, so it holds only what pickles."""

    base_url: str
    path: str

    def __call__(self, case_id: str, mode: str) -> None:
        running = server_running(self.base_url)
        load, cores = machine_load()
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": case_id, "mode": mode, "running": running, "load": load, "cores": cores}) + "\n")


@contextmanager
def _sampling(path: Path) -> Iterator[None]:
    """The sweep machine's load sampled into `path` at once and every `LOAD_EVERY_S` after, until the caller is done."""
    stop = threading.Event()

    def sample() -> None:
        while True:
            load, cores = machine_load()
            try:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"load": load, "cores": cores}) + "\n")
            except OSError:
                return
            if stop.wait(LOAD_EVERY_S):
                return

    sampler = threading.Thread(target=sample, name="mnemiq-sweep-load", daemon=True)
    sampler.start()
    try:
        yield
    finally:
        stop.set()
        sampler.join()


# --- the candidates ---------------------------------------------------------------------------------------------------


def _mean(values: Iterable[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return statistics.fmean(known) if known else None


def _tuning_records(cfg: SweepConfig, repeats: int) -> list[dict]:
    """The search's records, its start and its arms, each once: a grid that follows a screen shows the screen's start
    again."""
    seen, found = set(), []
    for round in _ROUNDS_OF[cfg.search.strategy]:
        for record in report(cfg, round, repeats=repeats, controls=False)[1]:
            key = (record["model"], record["mode"], record["round"], record["arm"])
            if key not in seen:
                seen.add(key)
                found.append(record)
    return found


def _p(record: dict, run: int) -> float:
    return next((pair["p"] for pair in record["pairs"] if pair["run"] == run), 1.0)


def _record(cfg: SweepConfig) -> dict:
    """The sweep's `sweep.json`, or an empty record before a run has started the sweep."""
    path = Path(cfg.sweep.out) / SWEEP
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _out_of_room(state: dict, model: str, mode: str) -> dict | None:
    """The room the record says `model` in `mode` ran out of, when its search stopped for want of room; else None."""
    room = state.get("room", {}).get(model, {}).get(mode, {})
    return room if room.get("stopped") == "no room" else None


def _judged(cfg: SweepConfig, records: Sequence[dict], state: dict) -> dict[tuple[str, str], dict]:
    """For each model and mode, its start's record and every arm compared with it, judged by the rule: each arm's mean
    right and prompt tokens, whether its p passes Holm in each run, and why it is not a candidate. Arms are sorted by
    mean right, the most first.

    A model and mode whose search the record (`state`) says stopped for want of room has no candidate. The arms that
    were running beside the start's first run when that was found, and finished, are judged all the same, and each
    says the search stopped there."""
    alpha, bounds = cfg.propose.alpha, cfg.propose
    judged: dict[tuple[str, str], dict] = {}
    for model in cfg.models:
        for mode in cfg.modes:
            mine = [record for record in records
                    if (record["model"], record["mode"]) == (model, mode) and record["control"] is None]
            start = next((record for record in mine if record["arm"] == "start"), None)
            arms = [record for record in mine if record["arm"] != "start"]
            runs = len(start["runs"]) if start is not None else max((len(record["runs"]) for record in arms), default=0)
            passed = [holm({record["arm"]: _p(record, run) for record in arms}, alpha) for run in range(1, runs + 1)]
            room = _out_of_room(state, model, mode)
            entries = []
            for record in arms:
                pairs = {pair["run"]: pair for pair in record["pairs"]}
                why = []
                if room is not None:
                    why.append(f"the search stopped in {mode} for want of room once the start's first run was in: the "
                               f"start missed {room.get('misses')} of the questions to tune on, fewer than an arm "
                               "must gain to win, so no arm there is a candidate, whatever it scored")
                if start is None or not start["valid"]:
                    why.append("the start is not valid in every run")
                if not record["valid"]:
                    why.append("not valid in every run")
                for run in range(1, runs + 1):
                    found = pairs.get(run)
                    if found is None:
                        why.append(f"run {run} has no pair with the start's")
                        continue
                    if found["only"] <= found["other"]:
                        why.append(f"run {run}: right alone on {found['only']} questions and wrong alone on "
                                   f"{found['other']}, so no better than the start")
                    elif record["arm"] not in passed[run - 1]:
                        why.append(f"run {run}: p {found['p']:.3g} does not pass Holm's correction over {len(arms)} "
                                   f"arms at alpha {alpha:g}")
                    if found["more_wrong"] > bounds.max_more_wrong:
                        why.append(f"run {run}: {found['more_wrong']} more wrong answers than the start, above "
                                   f"max_more_wrong {bounds.max_more_wrong}")
                    if found["more_made_up"] > bounds.max_more_made_up:
                        why.append(f"run {run}: {found['more_made_up']} more made-up answers than the start, above "
                                   f"max_more_made_up {bounds.max_more_made_up}")
                entries.append({
                    "label": record["label"], "arm": record["arm"], "round": record["round"],
                    "changes": record["changes"], "valid": record["valid"], "runs": record["runs"],
                    "pairs": record["pairs"], "mean_right": _mean(run["right"] for run in record["runs"]),
                    "prompt_tokens": _mean(run["prompt_tokens"] for run in record["runs"]),
                    "holm_passed": [record["arm"] in each for each in passed], "candidate": not why, "why_not": why,
                })
            entries.sort(key=lambda entry: (-(entry["mean_right"] or 0.0), entry["label"]))
            judged[(model, mode)] = {"start": start, "arms": entries, "compared": len(arms)}
    return judged


def candidates(cfg: SweepConfig, records: Sequence[dict],
               state: dict | None = None) -> dict[tuple[str, str], list[str]]:
    """For each model and mode, the labels of its candidate arms, the most right on average first. `records` are the
    search's, as `report` gives them for the screen and the grid; `state` is the sweep's record, by default its
    `sweep.json`."""
    return {key: [entry["label"] for entry in found["arms"] if entry["candidate"]]
            for key, found in _judged(cfg, records, _record(cfg) if state is None else state).items()}


def _top(found: dict) -> list[dict]:
    """The candidates tied on the most right answers on average."""
    chosen = [entry for entry in found["arms"] if entry["candidate"]]
    if not chosen:
        return []
    best = max(entry["mean_right"] for entry in chosen)
    return [entry for entry in chosen if entry["mean_right"] == best]


def _arm_of(model: str, mode: str, entry: dict) -> Arm:
    """The arm a judged entry was read from, as its round built it."""
    arm = arm_of(model, mode, entry["changes"], round=entry["round"])
    if arm.profile_id != entry["arm"]:
        raise SweepConfigError(f"the arm {entry['label']!r} of {model} in {mode} reads back as {arm.profile_id}, not "
                               f"{entry['arm']}: its folder was written under another config")
    return arm


# --- the timing round -------------------------------------------------------------------------------------------------


def _timed(cfg: SweepConfig, judged: dict[tuple[str, str], dict]) -> dict[tuple[str, str], list[Arm]]:
    """For each model and mode with a candidate, what the timing round runs: the start, then each candidate tied on the
    most right answers."""
    return {(model, mode): [arm_of(model, mode, {}, round="timing"),
                            *(replace(_arm_of(model, mode, entry), round="timing") for entry in _top(found))]
            for (model, mode), found in judged.items() if _top(found)}


def _reading(cfg: SweepConfig, arm: Arm, record: dict | None, timed: Sequence[str]) -> dict:
    """One timed run, with its checks. It is taken when it is valid, every case it timed was checked, the server ran
    nothing else before any case, and the machine's load stayed under its core count in every sample. A server that
    published no count for any case is hosted, and its time is indicative."""
    folder = arm_folder(cfg, arm, 1)
    checked = [line for line in _rows_so_far(folder / CHECKS) if line.get("mode") == arm.mode]
    samples = [*checked, *_rows_so_far(folder / LOADS)]
    running = [line.get("running") for line in checked]
    hosted = bool(checked) and all(value is None for value in running)
    busy = sum(1 for value in running if value)
    unread = 0 if hosted else sum(1 for value in running if value is None)
    loads = [(line.get("load"), line.get("cores")) for line in samples]
    over = sum(1 for load, cores in loads if load is None or cores is None or load >= cores)
    every_case = {line.get("id") for line in checked} >= set(timed)
    run = record["runs"][0] if record is not None else {}
    valid = record is not None and record["valid"]
    return {
        "label": arm.label, "arm": arm.profile_id, "valid": valid, "median_ms": run.get("median_ms"),
        "median_rows_ms": run.get("median_rows_ms"), "prompt_tokens": run.get("prompt_tokens"),
        "taken": valid and every_case and not busy and not unread and not over, "indicative": hosted,
        "checks": {"cases": len(checked), "server_busy": busy, "server_unread": unread, "load_samples": len(loads),
                   "load_at_or_over_cores": over,
                   "max_load": max((load for load, _ in loads if load is not None), default=None),
                   "cores": next((cores for _, cores in loads if cores is not None), None)},
    }


def _timing(cfg: SweepConfig, judged: dict[tuple[str, str], dict]) -> dict[tuple[str, str], dict]:
    """For each model and mode that was timed, its readings, whether every one was taken, and whether any is
    indicative."""
    records = report(cfg, "timing", controls=False)[1]
    parts = saved_split(cfg)
    timed = [case.id for case in (parts["tuning"] if parts is not None else [])[:TIMING_CASES]]
    found = {}
    for (model, mode), arms in _timed(cfg, judged).items():
        mine = {record["arm"]: record for record in records if (record["model"], record["mode"]) == (model, mode)}
        readings = [_reading(cfg, arm, mine.get(arm.profile_id), timed) for arm in arms]
        found[(model, mode)] = {"taken": all(reading["taken"] for reading in readings),
                                "indicative": any(reading["indicative"] for reading in readings), "readings": readings}
    return found


def _pick(found: dict, timing: dict | None) -> tuple[dict | None, str]:
    """The top candidate of one model and mode, and what decided it."""
    top = _top(found)
    if not top:
        return None, "no candidate"
    if len(top) == 1:
        return top[0], "the most right answers"
    readings = {reading["arm"]: reading for reading in (timing or {}).get("readings", [])}
    timed = all(readings.get(entry["arm"], {}).get("taken") and not readings[entry["arm"]]["indicative"]
                and readings[entry["arm"]]["median_ms"] is not None for entry in top)

    def order(entry: dict) -> tuple[float, float, str]:
        tokens = entry["prompt_tokens"]
        return (readings[entry["arm"]]["median_ms"] if timed else 0.0,
                tokens if tokens is not None else math.inf, entry["label"])

    first, second = sorted(top, key=order)[:2]
    if order(first)[0] < order(second)[0]:
        return first, "tied on right answers: the lower median time"
    if order(first)[1] < order(second)[1]:
        return first, "tied on right answers" + ("" if timed else ", with no time to compare") + ": fewer prompt tokens"
    return first, "tied on right answers, time and tokens: the first label"


def pick(cfg: SweepConfig, model: str, mode: str) -> Arm | None:
    """The top candidate of `model` in `mode`, as its round built it, or None when no arm is a candidate. A tie on mean
    right goes to the lower median time of a taken reading that is not indicative, when every tied arm has one, then to
    fewer prompt tokens (a mean over the runs that tuned it)."""
    judged = _judged(cfg, _tuning_records(cfg, decided_repeats(cfg)), _record(cfg))
    entry, _ = _pick(judged[(model, mode)], _timing(cfg, judged).get((model, mode)))
    return None if entry is None else _arm_of(model, mode, entry)


# --- the modes the start ran in --------------------------------------------------------------------------------------


def _ran(cfg: SweepConfig, state: dict) -> dict[str, tuple[list[str], dict[str, str]]]:
    """For each model, the modes whose start the sweep ran, in the config's order, and why the engine refused the start
    in each other one: the first round's plan recorded them in `starts`, and they are never asked of the engine again,
    whose environment or build may have moved since."""
    starts = state.get("starts")
    if not isinstance(starts, dict) or any(not isinstance(starts.get(model), dict) for model in cfg.models):
        raise SweepConfigError(f"{Path(cfg.sweep.out) / SWEEP} was written before this build kept which modes the "
                               "start ran in, so no proposal can be read from it: start a new sweep in another out "
                               "folder.")
    found = {}
    for model in cfg.models:
        known = starts[model]
        running = [mode for mode in cfg.modes if mode in known and known[mode] is None]
        found[model] = (running, {mode: known.get(mode) or "the first round's plan holds no start for it"
                                  for mode in cfg.modes if mode not in running})
    return found


def _joined(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _profile_wide(arms: tuple[Arm, ...] | None) -> dict | None:
    """The whole-profile changes from the start that the written profile makes, which every mode runs under; None when
    nothing is written. Arms merged mode by mode hold the same ones: their files share every profile-wide line."""
    if arms is None:
        return None
    return {name: value for name, value in arms[0].changes if _mode_of(name) is None}


def _unmeasured_there(modes: Sequence[str], wide: dict) -> str:
    """What a profile written with the whole-profile changes `wide` does in `modes`, which hold no section of it."""
    one = len(modes) == 1
    note = (f" The profile holds no section for {'that mode' if one else 'those modes'}, so "
            f"{'it keeps' if one else 'each keeps'} the engine's own per-mode settings, and runs under the profile's "
            "whole-profile settings")
    if not wide:
        return note + ", which change nothing from the start."
    changes = ", ".join(f"{name}={_show(value)}" for name, value in sorted(wide.items()))
    return note + (f", whose {'change' if len(wide) == 1 else 'changes'} from the start ({changes}) "
                   f"{'was' if len(wide) == 1 else 'were'} never measured there.")


def _refused_note(refused: dict[str, str], wide: dict | None) -> str:
    """What the reason says of the modes the engine refused the start in, or nothing. With a profile written (`wide`,
    its whole-profile changes from the start), each such mode keeps the engine's own per-mode settings, having no
    section of its own, and runs under the profile's whole-profile settings, whose changes were never measured
    there."""
    if not refused:
        return ""
    modes = list(refused)
    note = (f" The engine refused the start in {_joined(modes)} "
            f"({'; '.join(f'{mode}: {why}' for mode, why in refused.items())}), so nothing ran there.")
    return note if wide is None else note + _unmeasured_there(modes, wide)


def _unsearched_note(unsearched: Sequence[str], wide: dict | None) -> str:
    """What the reason says of the engine's modes the sweep did not search, or nothing: with a profile written, as of
    a mode the engine refused the start in."""
    if not unsearched or wide is None:
        return ""
    return f" The sweep did not search {_joined(unsearched)}." + _unmeasured_there(unsearched, wide)


def _refused_everywhere(refused: dict[str, str]) -> str:
    named = "; ".join(f"{mode}: {why}" for mode, why in refused.items())
    return f"Nothing is proposed: the engine refuses the start in every mode ({named}), so nothing ran."


# --- the confirm round ------------------------------------------------------------------------------------------------


def _confirming(cfg: SweepConfig, model: str, judged: dict, timing: dict,
                running: Sequence[str]) -> tuple[Arm | None, Arm | None, dict, list[str]]:
    """The assembled arm the confirm round measures for `model` (None when no mode has a candidate), the start it is
    measured against (None when no mode is recorded as having run), each in every mode of `running`, each of those
    modes' pick (None for the start) and what decided it, and the profile-wide settings the picks disagreed on."""
    picks = {mode: _pick(judged[(model, mode)], timing.get((model, mode))) for mode in running}
    if not running:
        return None, None, picks, []
    start = replace(arm_of(model, running[0], {}, round="confirm"), modes=tuple(running))
    if not any(entry for entry, _ in picks.values()):
        return None, start, picks, []
    arms = {mode: _arm_of(model, mode, entry) if entry else arm_of(model, mode, {})
            for mode, (entry, _) in picks.items()}
    assembled, conflicts = assembled_arm(cfg, model, arms, modes=running)
    return assembled, start, picks, conflicts


def after_search(sweep, until: str | None) -> None:
    """The rounds that follow a search, driven through `run`'s rounds (`sweep`: its `cfg`, `state`, `parts`, `job`,
    `drain`, `halted` and `save`), then the sweep's record says it has finished. Check mode has neither round, and a
    sweep whose start the engine refuses in every mode has nothing to time or confirm.

    The timing round runs, for each model and mode with a candidate, the start and each candidate tied on the most right
    answers, once each, alone, on the first `TIMING_CASES` cases to tune on. It first waits, a bounded while, for the
    machine's load to fall (`quiet_machine`). Before each case the worker checks the server and the machine
    (`TimingCheck`), and the machine's load is sampled while the run lasts. With `until` set to "timing", nothing
    follows it.

    The confirm round then runs, for each model with a candidate in any mode, the one profile assembled from each mode's
    pick and the start, each in every mode recorded as having run its start, on the held-out cases, once per repeat.
    """
    cfg = sweep.cfg
    if cfg.search.strategy != "check":
        running = {model: modes for model, (modes, _) in _ran(cfg, sweep.state).items()}
        if any(running.values()) and not _time_and_confirm(sweep, until, running):
            return
    sweep.state["finished"] = True
    sweep.save()


def _time_and_confirm(sweep, until: str | None, running: dict[str, list[str]]) -> bool:
    """The timing round, then (unless `until` is "timing") the confirm round; True when both ran to their end."""
    cfg = sweep.cfg
    repeats = decided_repeats(cfg, sweep.state)
    judged = _judged(cfg, _tuning_records(cfg, repeats), sweep.state)
    cases = sweep.parts["tuning"][:TIMING_CASES]
    timed = [sweep.job(arm, 1, cases=cases, before=TimingCheck(cfg.models[arm.model].llm_base_url,
                                                                str(arm_folder(cfg, arm, 1) / CHECKS)))
             for arms in _timed(cfg, judged).values() for arm in arms]
    if any(job.left() for job in timed):
        quiet_machine()
    for job in timed:
        running_now = job.left() > 0
        if running_now:
            job.folder.mkdir(parents=True, exist_ok=True)
        with _sampling(job.folder / LOADS) if running_now else nullcontext():
            sweep.drain(deque([job]), parallel=1)
        if sweep.halted:
            return False
    if until == "timing":
        return False
    timing = _timing(cfg, judged)
    jobs = []
    for model in cfg.models:
        assembled, start, _, _ = _confirming(cfg, model, judged, timing, running[model])
        if assembled is not None:
            jobs += [sweep.job(arm, repeat, cases=sweep.parts["held_out"])
                     for repeat in range(1, repeats + 1) for arm in (start, assembled)]
    sweep.drain(deque(jobs))
    return not sweep.halted


def _confirm_failure(cfg: SweepConfig, start: dict | None, arm: dict | None, runs: int) -> str | None:
    """Why the assembled arm fails in one mode of the confirm round, or None when it passes there."""
    bounds = cfg.propose
    if start is None or arm is None:
        return "the confirm round holds no rows for it"
    if not start["valid"]:
        return "the start is not valid there, so nothing can be confirmed against it"
    if not arm["valid"]:
        return "it is not valid in every run"
    pairs = {pair["run"]: pair for pair in arm["pairs"]}
    for run in range(1, runs + 1):
        found = pairs.get(run)
        if found is None:
            return f"its run {run} has no pair with the start's"
        if found["only"] < found["other"]:
            return f"in run {run} it got {found['other'] - found['only']} fewer right than the start"
        if found["more_wrong"] > bounds.max_more_wrong:
            return (f"in run {run} it gave {found['more_wrong']} more wrong answers than the start, above "
                    f"max_more_wrong {bounds.max_more_wrong}")
        if found["more_made_up"] > bounds.max_more_made_up:
            return (f"in run {run} it gave {found['more_made_up']} more made-up answers than the start, above "
                    f"max_more_made_up {bounds.max_more_made_up}")
    return None


# --- the files --------------------------------------------------------------------------------------------------------


def _renamed(path: Path, name: str) -> str:
    """The profiles file at `path`, its one profile renamed to `name`, and refused unless that is all that changed."""
    try:
        text = path.read_text(encoding="utf-8")
        raw = tomllib.loads(text)
        old, new = _toml(raw["default"]), _toml(name)
        raw["profiles"][raw["default"]]
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError):
        raise _Stale(f"{path}, the profile file of an arm that ran, is missing or cannot be read, so the profile "
                     "proposed cannot be checked against its rows") from None
    lines = []
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\n")
        if bare == f"default = {old}":
            line = f"default = {new}\n"
        elif bare == f"[profiles.{old}]":
            line = f"[profiles.{new}]\n"
        elif bare.startswith(f"[profiles.{old}.") and bare.endswith("]"):
            line = f"[profiles.{new}.{bare[len(f'[profiles.{old}.'):]}\n"
        lines.append(line)
    renamed = "".join(lines)
    if tomllib.loads(renamed) != {"default": name, "profiles": {name: raw["profiles"][raw["default"]]}}:
        raise _Stale(f"{path} could not be renamed to {name!r} without changing what it holds")
    return renamed


def _merged(texts: dict[str, str], name: str, modes: Sequence[str]) -> str:
    """One profiles file from each mode's own, each renamed to `name`: the profile-wide lines they all hold, then each
    mode's section from that mode's file, in the order of `modes`. Refused when the files disagree on a profile-wide
    line."""
    head = f"[profiles.{_toml(name)}."
    shared, sections = set(), []
    for mode in modes:
        lines = texts[mode].splitlines(keepends=True)
        starts = [number for number, line in enumerate(lines) if line.startswith(head)]
        if len(starts) != 1 or lines[starts[0]].rstrip("\n") != f"{head}{mode}]":
            raise SweepConfigError(f"the profile measured in {mode} does not hold that mode's section alone")
        shared.add("".join(lines[:starts[0]]))
        sections.append("".join(lines[starts[0]:]))
    if len(shared) != 1:
        raise SweepConfigError("the profiles measured mode by mode disagree on a profile-wide setting, so no one file "
                               "holds what each mode measured")
    merged = shared.pop() + "\n".join(sections)
    whole = tomllib.loads(merged)["profiles"][name]
    for mode in modes:
        own = tomllib.loads(texts[mode])["profiles"][name]
        if {key: value for key, value in whole.items() if key not in modes} | {mode: whole[mode]} != own:
            raise SweepConfigError(f"the merged profile does not hold what {mode} measured")
    return merged


def _versions(cfg: SweepConfig, path: Path, dialects: dict) -> dict[str | None, str]:
    """The version the engine gives the default profile of the file at `path`, for each deployment, as a run's runtime
    stamps it: the engine's own loader on the deployment's baseline, and its version function with the dialect of the
    deployment's own adapter (read once into `dialects`)."""
    found = {}
    for db_id, settings in baselines(cfg):
        try:
            profiles, default = load_profiles(settings.model_copy(update={"profiles_path": str(path)}))
        except ValueError as exc:  # the profile loader's refusals, with every key variable set (`_keys_set`)
            raise _Stale(f"the engine refuses the proposed profile, so its version cannot be recomputed: "
                         f"{scrub_credentials(str(exc), settings)}") from None
        if db_id not in dialects:
            try:
                adapter = source_adapter(settings, read_only=True)
            except Exception as exc:  # noqa: BLE001 -- an attach error can quote a DSN, so none is passed on as it is
                where = f"the deployment for {db_id!r}" if db_id is not None else "the deployment"
                said = redact.scrub_credentials(scrub_credentials(str(exc), settings))
                raise SweepConfigError(f"{where}'s source cannot be opened to read its dialect, which a profile's "
                                       f"version holds: {said}") from None
            try:
                dialects[db_id] = getattr(adapter, "dialect", "duckdb")
            finally:
                close = getattr(adapter, "close", None)
                if close is not None:
                    close()
        found[db_id] = profile_version(profiles[default].settings, dialects[db_id])
    return found


def _stamped(cfg: SweepConfig, arm: Arm, repeats: int, mode: str | None = None) -> dict[str | None, str]:
    """The version each deployment's rows of `arm` hold, over its runs (in `mode` alone, when given); refused unless
    each holds exactly one."""
    reads = {case.id: case.db_id if cfg.dataset.deployments is not None else None for case in load_questions(cfg)}
    held: dict[str | None, set] = defaultdict(set)
    for repeat in range(1, repeats + 1):
        for row in _rows_so_far(arm_folder(cfg, arm, repeat) / ROWS):
            if mode is None or row.get("mode") == mode:
                held[reads.get(row.get("id"))].add(row.get("profile_version"))
    if not held or any(len(versions) != 1 or None in versions for versions in held.values()):
        raise _Stale(f"the rows of {arm.label} in the {arm.round} round do not hold one version for each deployment, "
                     "so the profile proposed cannot be checked against them")
    return {db_id: next(iter(versions)) for db_id, versions in held.items()}


def _shown(versions: dict[str | None, str]) -> str | dict[str, str]:
    """A version for the evidence: one, with one deployment; else one for each db_id."""
    return versions[None] if list(versions) == [None] else {str(db_id): version for db_id, version in versions.items()}


def _agrees(recomputed: dict, stamped: dict) -> bool:
    return all(recomputed.get(db_id) == version for db_id, version in stamped.items())


_REFUSAL = ("the version the engine gives the profile file {what} is not the one its rows were stamped with: the file "
            "is not the profile that was measured, so it is not proposed")


@dataclass(frozen=True)
class _Draft:
    """A proposed profiles file, every check made on a draft of it before anything is written: its text, the version
    each deployment's engine gives it, and what the evidence says of it. No text when nothing is proposed."""

    text: str | None
    versions: dict
    shown: dict


def _draft(cfg: SweepConfig, model: str, arms: tuple[Arm, ...] | None, repeats: int, modes: Sequence[str]) -> _Draft:
    """`model`'s proposed profiles file, from the own files of `arms`, which ran: one arm in every mode of `modes`, or
    one arm for each, merged. Each arm's file is checked: the version the engine recomputes on it must be the one its
    rows were stamped with, and for one arm in every mode that is checked on the draft of the file to be written."""
    if arms is None:
        return _Draft(None, {}, {"profile": None, "measured": None, "profile_from": [], "version": None,
                                 "confirmed_arm_version": None})
    name = cfg.propose.profile or model
    texts = [(arm, _renamed(arm_folder(cfg, arm, 1) / PROFILE, name)) for arm in arms]
    whole = len(arms) == 1
    if whole and arms[0].modes != tuple(modes):
        raise SweepConfigError(f"{arms[0].label} ran in {', '.join(arms[0].modes)}, not in {', '.join(modes)}")
    text = texts[0][1] if whole else _merged({arm.mode: own for arm, own in texts}, name, modes)
    dialects: dict = {}
    measured: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="mnemiq-sweep-proposal-") as scratch:
        draft = Path(scratch) / PROFILES
        draft.write_text(text, encoding="utf-8")
        versions = _versions(cfg, draft, dialects)
        if whole:
            measured[arms[0].mode] = _stamped(cfg, arms[0], repeats)
            if not _agrees(versions, measured[arms[0].mode]):
                raise _Stale(_REFUSAL.format(what=f"of {arms[0].label} in the {arms[0].round} round"))
        for arm, own in texts if not whole else ():
            own_path = Path(scratch) / f"own-{arm.mode}.toml"
            own_path.write_text(own, encoding="utf-8")
            measured[arm.mode] = _stamped(cfg, arm, repeats, arm.mode)
            if not _agrees(_versions(cfg, own_path, dialects), measured[arm.mode]):
                raise _Stale(_REFUSAL.format(what=f"of {arm.label} in the {arm.round} round, in {arm.mode}"))
    shown = {"profile": name, "measured": "whole" if whole else "mode by mode",
             "profile_from": [{"round": arm.round, "arm": arm.profile_id, "modes": list(arm.modes)} for arm in arms],
             "version": _shown(versions), "confirmed_arm_version": _shown(measured[arms[0].mode]) if whole else None}
    if not whole:
        shown["measured_versions"] = {mode: _shown(each) for mode, each in measured.items()}
    return _Draft(text, versions, shown)


def _folder(cfg: SweepConfig, model: str) -> Path:
    return Path(cfg.sweep.out) / PROPOSAL / model


def _withdraw(cfg: SweepConfig) -> None:
    """Remove every proposal file an earlier `propose` left: after a refusal, none may look current."""
    for model in cfg.models:
        for name in (PROFILES, EVIDENCE):
            (_folder(cfg, model) / name).unlink(missing_ok=True)


def _write(cfg: SweepConfig, model: str, draft: _Draft, evidence: dict) -> None:
    """Write `model`'s two proposal files: the profile (or none, removing an earlier one) and then its evidence. The
    version is recomputed on the profile as written, and must be the draft's."""
    folder = _folder(cfg, model)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / EVIDENCE).unlink(missing_ok=True)  # until the profile beside it is written and checked
    if draft.text is None:
        (folder / PROFILES).unlink(missing_ok=True)
    else:
        _replace(folder / PROFILES, draft.text)
        if _versions(cfg, folder / PROFILES, {}) != draft.versions:
            raise _Stale(_REFUSAL.format(what="as written"))
    _write_json(folder / EVIDENCE, evidence)


# --- the proposal -----------------------------------------------------------------------------------------------------


def _finished(cfg: SweepConfig) -> dict:
    """The sweep's record, refused unless the sweep was run under this config and has finished."""
    record = Path(cfg.sweep.out) / SWEEP
    if not record.exists():
        raise SweepConfigError(f"{cfg.sweep.out} holds no sweep ({SWEEP}): run the sweep before proposing from it")
    state = json.loads(record.read_text(encoding="utf-8"))
    if state.get("fingerprint") != fingerprint(cfg):
        raise SweepConfigError(f"{record}: this sweep was run under another config. The rule a proposal applies is "
                               "part of the config, so it is read under the config the sweep ran under, or not at "
                               "all.")
    if not state.get("finished"):
        stop = f" (it stopped: {state['stopped']})" if state.get("stopped") in ("budget", "endpoint", "worker") else ""
        raise _Stale(f"{record}: the sweep has not finished its rounds{stop}. Run it again: it resumes where it "
                     "stopped.")
    try:
        saved_split(cfg)
    except SweepConfigError as exc:  # a split that is not the one the sweep drew
        raise _Stale(str(exc)) from None
    return state


def _keys_set(cfg: SweepConfig, models: Collection[str]) -> None:
    """Refuse unless each key variable `models` name is set: the engine's loader reads it to recompute a profile's
    version. Named, never quoted."""
    if unset := unset_keys(cfg, models):
        raise SweepConfigError(f"a key variable the proposal needs is not set: {', '.join(unset)}. The engine reads it "
                               "to recompute each profile's version; set it and propose again.")


def _build_warning(state: dict, packaged: dict) -> list[str]:
    """A warning when the engine packaging the proposal is not the one the arms ran on: the rows were measured on the
    latter, and the version check still guards the file."""
    ran = str((state.get("engine_build") or {}).get("source_sha"))
    if ran == packaged["source_sha"]:
        return []
    return [f"The sweep's arms ran on engine build {ran[:12]} (by source), and this proposal is packaged by build "
            f"{packaged['source_sha'][:12]}. The rows were measured on the first. The profile's version is recomputed "
            "by this one and must still be the one the rows were stamped with, so a change to what the profile "
            "resolves to is refused."]


def _head(cfg: SweepConfig, model: str, state: dict, repeats: int, packaged: dict) -> dict:
    """What every proposal's evidence opens with: the sweep, who asked, where, on what, and by which rule."""
    parts = saved_split(cfg)
    endpoints = {name: without_credentials(value) if isinstance(value, str) and name.endswith("_url") else value
                 for name, value in _endpoint(cfg, model).items() if name in ENDPOINT_FACTS}
    rule = {"strategy": cfg.search.strategy, "alpha": cfg.propose.alpha, "max_more_wrong": cfg.propose.max_more_wrong,
            "max_more_made_up": cfg.propose.max_more_made_up, "max_errors": cfg.sweep.max_errors, "repeats": repeats,
            "repeats_decided": isinstance(cfg.sweep.repeats, int) or "repeats" in state}
    if cfg.search.strategy != "check":
        rule["screen_alpha"] = cfg.search.screen_alpha
    return {
        "model": model,
        "sweep": {"name": cfg.sweep.name, "fingerprint": state["fingerprint"], "engine_build": state["engine_build"],
                  "split_seed": cfg.dataset.split_seed},
        "packaged_by": packaged,
        "identity": cfg.dataset.identity.model_dump(),
        "endpoints": endpoints,
        "split": {part: {"cases": len(parts[part]), "answerable": sum(case.answerable for case in parts[part])}
                  for part in PARTS},
        "rule": rule,
        "unpinned_sources": unpinned_sources(baselines(cfg)),
    }


def _held_out_note(cfg: SweepConfig) -> str:
    """What the held-out part can show, said whatever it showed."""
    answerable = sum(case.answerable for case in saved_split(cfg)["held_out"])
    gain = detectable_gain(answerable, 1, cfg.propose.alpha) if answerable else None
    if gain is None:
        return (f"The held-out part holds {answerable} answerable questions: it can show harm, but no gain on so few "
                f"would pass the exact McNemar test at alpha {cfg.propose.alpha:g}.")
    return (f"The held-out part holds {answerable} answerable questions: it can show harm, and a gain only of at least "
            f"{gain} more right with none lost.")


def _unmeasured(run: dict, max_errors: int) -> str:
    counted = sum(run[name] for name in ("right", "wrong", "made_up", "declined", "failed", "errors", "ungraded",
                                         "unjudged"))
    if counted < run["cases"]:
        return f"run {run['run']} holds {counted} of its {run['cases']} cases"
    missed = run["errors"] + run["ungraded"] + run["unjudged"]
    if missed > max_errors:
        return f"run {run['run']} did not measure {missed} cases, more than max_errors {max_errors}"
    return f"run {run['run']} is not valid"


def _check(cfg: SweepConfig, model: str, records: Sequence[dict], running: Sequence[str],
           refused: dict[str, str]) -> dict:
    """Check mode's verdict for `model`, over the modes recorded as having run its start: the start, the defaults, or
    nothing."""
    bounds = cfg.propose
    numbers, invalid, harm = {}, [], []
    valid = {"start", "defaults"} if running else set()
    for mode in running:
        found = {arm: next((record for record in records if (record["model"], record["mode"], record["arm"])
                            == (model, mode, arm)), None) for arm in ("start", "defaults")}
        for arm, record in found.items():
            if record is None:
                invalid.append(f"the {arm} has no rows in {mode}")
            elif not record["valid"]:
                why = "; ".join(_unmeasured(run, cfg.sweep.max_errors) for run in record["runs"] if not run["valid"])
                invalid.append(f"the {arm} is not valid in {mode} ({why})")
            if record is None or not record["valid"]:
                valid.discard(arm)
        if None in found.values():
            continue
        apart = [{"run": start["run"], "more_wrong": start["wrong"] - defaults["wrong"],
                  "more_made_up": start["made_up"] - defaults["made_up"]}
                 for start, defaults in zip(found["start"]["runs"], found["defaults"]["runs"], strict=True)]
        harm += [f"in run {each['run']} of {mode} the start gives {each['more_wrong']} more wrong and "
                 f"{each['more_made_up']} more made-up answers than the defaults"
                 for each in apart if each["more_wrong"] > bounds.max_more_wrong
                 or each["more_made_up"] > bounds.max_more_made_up]
        numbers[mode] = {"start": found["start"]["runs"], "defaults": found["defaults"]["runs"],
                         "start_against_defaults": apart}
    if not running:
        rule, why = None, _refused_everywhere(refused)
    elif invalid:
        rule, why = None, ("Nothing is proposed: " + "; ".join(invalid) + ". Check mode proposes only between two "
                           "arms valid in every run.")
    elif harm:
        rule, why = "defaults", ("The defaults are proposed: " + "; ".join(harm) + ", beyond what [propose] allows "
                                 f"(max_more_wrong {bounds.max_more_wrong}, "
                                 f"max_more_made_up {bounds.max_more_made_up}).")
    else:
        rule, why = "start", ("The start is proposed: in every mode and run its wrong and made-up answers exceed the "
                              "defaults' by no more than [propose] allows. Both arms' right answers are in the "
                              "evidence.")
    return {"rule": rule, "why": why, "check": numbers, "valid": valid, "start_valid": "start" in valid}


def _check_arms(cfg: SweepConfig, model: str, profile_id: str, modes: Sequence[str]) -> tuple[Arm, ...]:
    return tuple(arm for mode in modes for arm in check_arms(cfg)
                 if (arm.model, arm.mode, arm.profile_id) == (model, mode, profile_id))


def _first_starts(cfg: SweepConfig, model: str, modes: Sequence[str]) -> tuple[Arm, ...]:
    """The start of `model` in each of `modes`, as the first round ran it."""
    return tuple(arm_of(model, mode, {}, round=_first_round(cfg)) for mode in modes)


def _search(cfg: SweepConfig, model: str, state: dict, judged: dict, timing: dict, confirm: Sequence[dict],
            repeats: int, running: list[str], refused: dict[str, str]) -> dict:
    """The rule's verdict for `model` in a sweep that searched, over the modes recorded as having run its start, with
    what its evidence shows. Where the rule would fall back to the start and the start is not valid in every run of
    every such mode, nothing is proposed: no arm could be compared with it, and it was not measured either."""
    verdict = _searched(cfg, model, state, judged, timing, confirm, repeats, running, refused)
    if verdict["rule"] != "start" or verdict["start_valid"]:
        return verdict
    said = []
    for mode in running:
        start = judged[(model, mode)]["start"]
        if start is None:
            said.append(f"it has no rows in {mode}")
        elif not start["valid"]:
            runs = "; ".join(_unmeasured(run, cfg.sweep.max_errors) for run in start["runs"] if not run["valid"])
            said.append(f"in {mode}, {runs}")
    failed = ("The one profile assembled from each mode's pick failed on the held-out part in "
              f"{verdict['failed_mode']}, and the start it falls back to is not valid. "
              if verdict["confirmed"] is False else "")
    return verdict | {"rule": None, "rule_arms": None, "confirmed": None,
                      "why": f"{failed}Nothing is proposed: the start is not valid in every run "
                             f"({'; '.join(said)}), so nothing can be compared with it, and a start that was not "
                             "measured is not proposed."}


def _searched(cfg: SweepConfig, model: str, state: dict, judged: dict, timing: dict, confirm: Sequence[dict],
              repeats: int, running: list[str], refused: dict[str, str]) -> dict:
    """`_search`'s verdict before it looks at whether the start it may fall back to is valid."""
    room = state.get("room", {}).get(model, {})
    no_room = [mode for mode in running if room.get(mode, {}).get("stopped") == "no room"]
    assembled, start, picks, conflicts = _confirming(cfg, model, judged, timing, running)
    verdict = {
        "picks": {mode: entry["label"] if entry else "start" for mode, (entry, _) in picks.items()},
        "picked_by": {mode: how for mode, (entry, how) in picks.items() if entry},
        "conflicts": conflicts, "failed_mode": None, "confirm": None, "numbers": None, "room": room,
        "holm": {mode: judged[(model, mode)]["compared"] for mode in running},
        "candidates": {mode: {"compared": judged[(model, mode)]["compared"],
                              "start": (judged[(model, mode)]["start"] or {}).get("runs"),
                              "arms": judged[(model, mode)]["arms"]} for mode in running},
        "timing": {mode: timing[(model, mode)] for mode in running if (model, mode) in timing},
        # The start's own file: the confirm round's, measured in every mode, when that round ran.
        "start_arms": (start,) if assembled is not None else _first_starts(cfg, model, running),
        "start_valid": bool(running) and all((judged[(model, mode)]["start"] or {}).get("valid") for mode in running),
        "assembled": None, "assembled_valid": False,
    }
    if not running:
        return verdict | {"rule": None, "rule_arms": None, "confirmed": None, "why": _refused_everywhere(refused)}
    if len(no_room) == len(running):
        return verdict | {"rule": "start", "rule_arms": verdict["start_arms"], "confirmed": None,
                          "why": "There was no room to search in any mode: the start missed fewer of the questions to "
                                 "tune on than an arm must gain to win, so no arm could beat it, and the start is "
                                 "proposed. Check mode compares it with the engine's defaults for harm."}
    if assembled is None:
        unmeasured = [mode for mode in running if not (judged[(model, mode)]["start"] or {}).get("valid")]
        said = [f"{mode}: the start is not valid, so no arm could be compared with it" for mode in unmeasured]
        said += [f"{mode}: {judged[(model, mode)]['compared']} arms compared with the start" for mode in running
                 if mode not in no_room and mode not in unmeasured]
        said += [f"{mode}: no room to search" for mode in no_room]
        said += [f"{mode}: the engine refuses the start" for mode in refused]
        return verdict | {"rule": "start", "rule_arms": verdict["start_arms"], "confirmed": None,
                          "why": f"No arm beat the start beyond noise ({'; '.join(said)}; exact McNemar test, Holm's "
                                 f"correction at alpha {cfg.propose.alpha:g}), so the start is proposed."}
    numbers, failures, valid = {}, {}, []
    for mode in running:
        mine = {arm: next((record for record in confirm if (record["model"], record["mode"], record["arm"])
                           == (model, mode, arm)), None) for arm in ("start", assembled.profile_id)}
        held_start, held_arm = mine["start"], mine[assembled.profile_id]
        valid.append(held_arm is not None and held_arm["valid"])
        entry = picks[mode][0]
        numbers[mode] = {
            "held_out": {"start": held_start and held_start["runs"], "arm": held_arm and held_arm["runs"],
                         "pairs": held_arm and held_arm["pairs"]},
            "tuning_selected_on": {"pick": entry["label"] if entry else "start",
                                   "start": (judged[(model, mode)]["start"] or {}).get("runs"),
                                   "arm": entry["runs"] if entry else None},
        }
        if (failure := _confirm_failure(cfg, held_start, held_arm, repeats)) is not None:
            failures[mode] = failure
    failed = next((mode for mode in running if mode in failures), None)
    same = not assembled.changes
    verdict |= {"numbers": numbers, "failed_mode": failed, "assembled": assembled, "assembled_valid": all(valid),
                "confirm": {"label": assembled.label, "arm": assembled.profile_id, "changes": dict(assembled.changes),
                            "modes": list(assembled.modes), "same_as_start": same, "passed": failed is None}}
    shown = ", ".join(f"{mode}: {label}" for mode, label in verdict["picks"].items())
    lost = (" Every pick's change was lost to a conflict between the modes, so the profile holds the start's settings."
            if same else "")
    if failed is None:
        return verdict | {"rule": assembled.label, "rule_arms": (assembled,), "confirmed": True,
                          "why": f"{assembled.label} is the one profile assembled from each mode's pick ({shown}), and "
                                 "it was confirmed on the held-out part in every mode and run: valid, with at least "
                                 "the start's right answers and no more wrong or made-up answers than [propose] "
                                 f"allows.{lost}"}
    return verdict | {"rule": "start", "rule_arms": (start,), "confirmed": False,
                      "why": f"The one profile assembled from each mode's pick ({shown}) failed on the held-out part "
                             f"in {failed}: {failures[failed]}. No other candidate is tried, since that would make the "
                             f"held-out part part of the search, so the start is proposed.{lost}"}


def _chosen_arms(cfg: SweepConfig, model: str, verdict: dict, chosen: str,
                 running: Sequence[str]) -> tuple[tuple[Arm, ...], bool]:
    """The arms whose own files make up a person's choice for `model`, and whether that choice is the confirmed arm.
    Refused unless it names an arm that ran and is valid in every run."""
    if cfg.search.strategy == "check":
        if chosen not in ("start", "defaults") or chosen not in verdict["valid"]:
            raise SweepConfigError(f"{chosen!r} is not an arm of {model} valid in every run: check mode's are "
                                   f"{', '.join(sorted(verdict['valid'])) or 'none'}")
        return _check_arms(cfg, model, chosen, running), False
    if chosen == "start":
        if not verdict["start_valid"]:
            raise SweepConfigError(f"the start of {model} is not valid in every run, so a person cannot choose it")
        return verdict["start_arms"], False
    assembled = verdict["assembled"]
    if assembled is not None and chosen == assembled.label and verdict["assembled_valid"]:
        return (assembled,), bool(verdict["confirmed"])   # it ran in every mode, whether or not it passed
    arms, valid = [], []
    for mode in running:
        entries = verdict["candidates"][mode]["arms"]
        valid += [entry["label"] for entry in entries if entry["valid"]]
        entry = next((entry for entry in entries if entry["label"] == chosen), None)
        if entry is not None and not entry["valid"]:
            raise SweepConfigError(f"{chosen!r} is not valid in every run of {model} in {mode}")
        arms.append(_arm_of(model, mode, entry) if entry is not None else arm_of(model, mode, {},
                                                                                  round=_first_round(cfg)))
    if all(not arm.changes for arm in arms):
        raise SweepConfigError(f"{chosen!r} is not an arm of {model} that ran and is valid in every run: those are "
                               f"start, {', '.join(sorted(set(valid))) or 'no other'}")
    return tuple(arms), False


def propose(cfg: SweepConfig, chosen: str | None = None) -> dict:
    """Apply the rule to a finished sweep and write, for each model, `<out>/proposal/<model>/profiles.toml` (unless
    nothing is proposed) and `evidence.json`. Both hold no key: a profiles file names the variable that holds one.

    `chosen` is a person's pick: an arm's label (or "start", or check mode's "defaults") that ran and is valid in every
    run, for each model whose start ran in some mode. Its file is written instead, and the evidence says so, whether it
    is the arm the confirm round confirmed, and what the rule picked. A model whose start ran in no mode has nothing
    proposed, and says why.

    Refused, leaving an earlier proposal as it was, for a sweep run under another config, a record that does not say
    which modes the start ran in, a key variable that is not set, or a choice that names no arm that ran and is valid.
    Refused, removing an earlier proposal, when the sweep has not finished, its split is not the one drawn, or an arm's
    file cannot be read back as that arm or is not the profile its rows were stamped with; and the files are removed
    when writing them fails, however.
    A proposal packaged by another engine build than the one the arms ran on is written with a warning, its version
    still checked. Returns, for each model, what was chosen and by whom, the rule's pick, each mode's pick, the
    conflicts, the mode that failed the confirm round, how the profile was measured, why, any warnings, and the files
    written.

    It holds the out folder's lock, as a run does, so a proposal made while the sweep runs, or beside another
    proposal, is refused at once and changes nothing.
    """
    out = Path(cfg.sweep.out)
    if not out.is_dir():
        _finished(cfg)  # refuses: no sweep has run in that folder
    busy = SweepConfigError(f"{out} is in use by a run or another proposal of this sweep; propose again once it has "
                            "ended")
    with _sole(out / OUT_LOCK, busy):
        return _propose(cfg, chosen)


def _propose(cfg: SweepConfig, chosen: str | None) -> dict:
    """`propose` itself, with the out folder's lock held."""
    try:
        drafts = _drafts(cfg, chosen)
    except _Stale:
        _withdraw(cfg)
        raise
    results = {}
    try:
        for model, (draft, evidence) in drafts.items():   # every check passed: write each model's two files
            _write(cfg, model, draft, evidence)
            folder = _folder(cfg, model)
            results[model] = {key: evidence[key] for key in ("chosen", "chosen_by", "rule_pick", "confirmed", "picks",
                                                             "conflicts", "failed_mode", "measured", "why", "warnings")}
            results[model] |= {"profiles_file": str(folder / PROFILES) if draft.text is not None else None,
                               "evidence_file": str(folder / EVIDENCE)}
    except BaseException:
        _withdraw(cfg)
        raise
    return results


def _drafts(cfg: SweepConfig, chosen: str | None) -> dict[str, tuple[_Draft, dict]]:
    """Each model's proposal, its profile drafted and checked and its evidence put together, nothing yet written."""
    state = _finished(cfg)
    ran = _ran(cfg, state)
    _keys_set(cfg, [model for model, (running, _) in ran.items() if running])
    if chosen and not any(running for running, _ in ran.values()):
        raise SweepConfigError(f"{chosen!r} cannot be chosen: the start ran in no mode of any model")
    packaged = engine_build()
    warnings = _build_warning(state, packaged)
    for warning in warnings:
        _say(warning)
    repeats = decided_repeats(cfg, state) if any(running for running, _ in ran.values()) else 1
    try:
        verdicts = _verdicts(cfg, state, ran, repeats)
    except ArmFileError as exc:  # an arm's own file, read back by the report, that is not the arm its rows were of
        raise _Stale(str(exc)) from None
    picked = {model: _chosen_arms(cfg, model, verdicts[model], chosen, running) if chosen and running else None
              for model, (running, _) in ran.items()}
    unsearched = [mode for mode in MODES if mode not in cfg.modes]
    drafts = {}
    for model, verdict in verdicts.items():
        running, refused = ran[model]
        if picked[model] is not None:
            arms, confirmed = picked[model]
            why = (f"A person chose {chosen}{' (the arm confirmed)' if confirmed else ''}. The rule's own pick is "
                   f"{verdict['rule']}: {verdict['why']}")
        else:
            arms, confirmed, why = verdict["rule_arms"], bool(verdict["confirmed"]), verdict["why"]
            if chosen:
                why += f" A person's choice, {chosen}, does not apply here, since nothing ran."
        wide = _profile_wide(arms)
        evidence = _head(cfg, model, state, repeats, packaged) | {
            "start_valid": verdict["start_valid"],
            "chosen": chosen if picked[model] is not None else verdict["rule"],
            "chosen_by": "person" if picked[model] is not None else "rule",
            "rule_pick": verdict["rule"], "confirmed": confirmed,
            "why": why + (_refused_note(refused, wide) + _unsearched_note(unsearched, wide) if running else ""),
            "warnings": warnings, "picks": verdict["picks"], "conflicts": verdict["conflicts"],
            "failed_mode": verdict["failed_mode"],
            "refused_modes": {mode: {"why": reason, "unmeasured_profile_wide": wide or {}}
                              for mode, reason in refused.items()},
            "unsearched_modes": {mode: {"unmeasured_profile_wide": wide or {}} for mode in unsearched},
        }
        if cfg.search.strategy == "check":
            evidence["check"] = verdict["check"]
        else:
            evidence |= {key: verdict[key] for key in ("room", "holm", "candidates", "timing", "picked_by", "confirm",
                                                        "numbers")}
            evidence["held_out_note"] = _held_out_note(cfg)
        draft = _draft(cfg, model, arms, repeats, running)
        drafts[model] = (draft, evidence | draft.shown)
    return drafts


def _verdicts(cfg: SweepConfig, state: dict, ran: dict, repeats: int) -> dict[str, dict]:
    """The rule's verdict for each model, read from the rows of the modes recorded as having run its start."""
    if cfg.search.strategy == "check":
        records = report(cfg, "check", repeats=repeats, controls=False)[1]
        verdicts = {model: _check(cfg, model, records, *ran[model]) for model in cfg.models}
        for model, verdict in verdicts.items():
            arms = _check_arms(cfg, model, verdict["rule"], ran[model][0]) if verdict["rule"] else None
            verdict |= {"rule_arms": arms, "confirmed": None, "picks": {}, "conflicts": [], "failed_mode": None}
        return verdicts
    judged = _judged(cfg, _tuning_records(cfg, repeats), state)
    timing = _timing(cfg, judged)
    confirm = report(cfg, "confirm", repeats=repeats, controls=False)[1]
    return {model: _search(cfg, model, state, judged, timing, confirm, repeats, *ran[model]) for model in cfg.models}
