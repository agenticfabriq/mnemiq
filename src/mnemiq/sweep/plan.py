"""`plan`: what a sweep will run, what it will cost, and the smallest gain it could show, said before anything runs.

Nothing here calls a model or opens a source. The arms are the ones the engine accepts (each is loaded by the engine's
own profile loader); the grid is counted at its worst case, every value whose screen arm runs kept. Time comes from the
config's `seconds_per_question` until a run has measured it, and money from the models' prices and the tokens a run has
measured, so before one there is no dollar figure rather than a guess.

The smallest detectable gain is set beside the room's upper bound: the most an arm can gain is the questions the start
gets wrong, and before the start has run the tuning part's answerable questions bound that from above. When the gain
needs more than that, no arm can win, and check mode is the useful run.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, replace

from mnemiq.config import Settings
from mnemiq.sweep.arms import Arm, baselines, check_arms, grid_arms, refusal_under, screen_arms
from mnemiq.sweep.config import _FILE_SOURCES, SweepConfig, _source_specs, load_questions
from mnemiq.sweep.split import split
from mnemiq.sweep.stats import detectable_gain

# The timing round runs each finalist once, alone. How many finalists there are is known only after the grid; `plan`
# counts the start and one finalist for each model and mode, and says so.
TIMING_ARMS_PER_MODE = 2
_UNKNOWN = "unknown until the first questions run"
# The rounds that compare arms on the tuning part, by strategy; timing and confirm follow a search.
_ROUNDS_OF = {"screen-then-grid": ("screen", "grid"), "grid": ("grid",), "check": ("check",)}


@dataclass(frozen=True)
class Estimate:
    """What `render` says, as numbers.

    `arms` counts each round's arms that run, refusals left out; the confirm round counts the assembled profile and
    the start once in each mode, and timing counts `TIMING_ARMS_PER_MODE`. `case_runs`, `hours` and `dollars` are
    keyed by how many times each arm runs: one and two when the first round decides, the configured count otherwise.
    `detectable` and `compared` are per model and mode, for a sweep that searches.
    """

    arms: dict[str, int]
    case_runs: dict[int, int]
    hours: dict[int, float] | None
    dollars: dict[int, float] | None
    detectable: dict[tuple[str, str], int | None]
    compared: dict[tuple[str, str], int]
    room_bound: int
    store_bytes: dict[str, int | None]
    unpinned_sources: list[str]
    refused: list[tuple[Arm, str]]
    rounds: dict[tuple[str, str], dict[str, list[Arm]]]
    parts: dict[str, tuple[int, int]]  # tuning and held_out: (cases, answerable)


def _tried(arms: list[Arm]) -> dict[str, list[object]]:
    """The values one-change arms try, by setting."""
    tried: dict[str, list[object]] = {}
    for arm in arms:
        ((name, value),) = arm.changes
        tried.setdefault(name, []).append(value)
    return tried


def _rounds(cfg: SweepConfig, model: str, mode: str, screen: list[Arm],
            deployments: list[tuple[str | None, Settings]]) -> tuple[dict[str, list[Arm]], list[tuple[Arm, str]]]:
    """The arms each round runs for `model` in `mode`, the start first, and the arms the engine refuses.

    Every arm is compared with the start, so when the engine refuses the start nothing else runs. The screen's grid is
    counted at its worst case: every value whose screen arm runs, kept. A value whose screen arm the engine refuses
    never ran, so it cannot be kept.
    """
    refused: list[tuple[Arm, str]] = []

    def runs(arm: Arm) -> bool:
        why = refusal_under(cfg, arm, deployments)
        if why is not None:
            refused.append((arm, why))
        return why is None

    strategy = cfg.search.strategy
    if strategy == "check":
        start, *others = [a for a in check_arms(cfg) if (a.model, a.mode) == (model, mode)]
    else:
        start, *others = [a for a in screen if (a.model, a.mode) == (model, mode)]
    if not runs(start):
        return {name: [] for name in _ROUNDS_OF[strategy]}, refused
    if strategy == "check":
        return {"check": [start, *(arm for arm in others if runs(arm))]}, refused
    if strategy == "grid":
        # No screen: every combination of the space's values is tried, each checked by the engine on its own.
        combined = [replace(arm, round="grid") for arm in grid_arms(cfg, {(model, mode): _tried(others)})]
        return {"grid": [replace(start, round="grid"), *(arm for arm in combined if runs(arm))]}, refused
    screened = [start, *(arm for arm in others if runs(arm))]
    grid = [arm for arm in grid_arms(cfg, {(model, mode): _tried(screened[1:])}) if arm.round == "grid"]
    return {"screen": screened, "grid": [arm for arm in grid if runs(arm)]}, refused


def _questions_asked(cfg: SweepConfig, rounds: dict[str, list[Arm]], tuning: int, runs: int) -> tuple[int, int]:
    """Questions asked for one model and mode with each arm run `runs` times: (in parallel, alone). When the first
    round decides the repeats, its start runs twice either way."""
    decided = cfg.sweep.repeats == "auto"
    parallel = alone = 0
    for name, arms in rounds.items():
        if not arms:
            continue
        first = name in ("check", "screen") or cfg.search.strategy == "grid"
        count = 2 + (len(arms) - 1) * runs if first and decided else len(arms) * runs
        parallel += count * tuning
    if cfg.search.strategy != "check" and _searches(rounds):
        alone += TIMING_ARMS_PER_MODE * tuning
    return parallel, alone


def _searches(rounds: dict[str, list[Arm]]) -> bool:
    return any(arm.changes for arms in rounds.values() for arm in arms)


def estimate(cfg: SweepConfig, *, tokens: dict[str, tuple[float, float]] | None = None) -> Estimate:
    """The numbers `render` shows. `tokens` is what a run measured, per model: the mean tokens in and out of one question."""
    parts = split(cfg, load_questions(cfg))
    tuning, held_out = parts["tuning"], parts["held_out"]
    deployments = baselines(cfg)
    screen = [] if cfg.search.strategy == "check" else screen_arms(cfg)
    runs_each = (1, 2) if cfg.sweep.repeats == "auto" else (cfg.sweep.repeats,)

    rounds: dict[tuple[str, str], dict[str, list[Arm]]] = {}
    refused: list[tuple[Arm, str]] = []
    for model in cfg.models:
        for mode in cfg.modes:
            rounds[(model, mode)], skipped = _rounds(cfg, model, mode, screen, deployments)
            refused += skipped

    by_model: dict[str, dict[int, tuple[int, int]]] = {}
    for model in cfg.models:
        own = {mode: rounds[(model, mode)] for mode in cfg.modes}
        confirming = [mode for mode, r in own.items() if _searches(r)]
        totals = {}
        for runs in runs_each:
            parallel = alone = 0
            for r in own.values():
                p, a = _questions_asked(cfg, r, len(tuning), runs)
                parallel, alone = parallel + p, alone + a
            if confirming:
                # The assembled profile and the start, on the held-out part, in each mode the start runs in.
                running = [mode for mode, r in own.items() if any(r.values())]
                parallel += 2 * len(running) * len(held_out) * runs
            totals[runs] = (parallel, alone)
        by_model[model] = totals

    case_runs = {runs: sum(sum(by_model[m][runs]) for m in cfg.models) for runs in runs_each}
    hours = None
    if cfg.search.seconds_per_question is not None:
        seconds = cfg.search.seconds_per_question
        hours = {runs: sum(by_model[m][runs][0] / cfg.machine.parallel + by_model[m][runs][1] for m in cfg.models)
                 * seconds / 3600 for runs in runs_each}
    dollars = None
    if tokens is not None and all(_priced(cfg, m, tokens) for m in cfg.models):
        dollars = {runs: sum(sum(by_model[m][runs]) * _per_question(cfg, m, tokens) for m in cfg.models)
                   for runs in runs_each}

    arms = Counter()
    for (model, mode), r in rounds.items():
        for name, kept in r.items():
            arms[name] += len(kept)
        if cfg.search.strategy != "check" and _searches(r):
            arms["timing"] += TIMING_ARMS_PER_MODE
    for model in cfg.models:
        if any(_searches(rounds[(model, mode)]) for mode in cfg.modes):
            arms["confirm"] += 2 * sum(1 for mode in cfg.modes if any(rounds[(model, mode)].values()))

    room_bound = sum(case.answerable for case in tuning)
    compared: dict[tuple[str, str], int] = {}
    detectable: dict[tuple[str, str], int | None] = {}
    if cfg.search.strategy != "check":
        first = "grid" if cfg.search.strategy == "grid" else "screen"
        for key, r in rounds.items():
            if count := sum(1 for arm in r[first] if arm.changes):
                compared[key] = count
                detectable[key] = detectable_gain(room_bound, count, cfg.propose.alpha)

    store_bytes: dict[str, int | None] = {}
    unpinned: list[str] = []
    for db_id, settings in deployments:
        store = os.path.abspath(settings.store_path)
        store_bytes[store] = os.path.getsize(store) if os.path.isfile(store) else None
        for spec in _source_specs(settings):
            if spec.kind not in _FILE_SOURCES:
                unpinned.append(f"{spec.id} ({spec.kind})" if db_id is None else f"{db_id}: {spec.id} ({spec.kind})")

    return Estimate(
        arms=dict(arms), case_runs=case_runs, hours=hours, dollars=dollars, detectable=detectable, compared=compared,
        room_bound=room_bound, store_bytes=store_bytes, unpinned_sources=unpinned, refused=refused, rounds=rounds,
        parts={name: (len(part), sum(case.answerable for case in part)) for name, part in parts.items()})


def _priced(cfg: SweepConfig, model: str, tokens: dict[str, tuple[float, float]]) -> bool:
    entry = cfg.models[model]
    return model in tokens and entry.price_in_per_mtok is not None and entry.price_out_per_mtok is not None


def _per_question(cfg: SweepConfig, model: str, tokens: dict[str, tuple[float, float]]) -> float:
    entry = cfg.models[model]
    tokens_in, tokens_out = tokens[model]
    return (tokens_in * entry.price_in_per_mtok + tokens_out * entry.price_out_per_mtok) / 1_000_000


def _runs(runs: int) -> str:
    return {1: "one run per arm", 2: "two runs per arm"}.get(runs, f"{runs} runs per arm")


def _size(count: int) -> str:
    size, unit = float(count), "bytes"
    for larger in ("KB", "MB", "GB", "TB"):
        if size < 1024:
            break
        size, unit = size / 1024, larger
    return f"{count} bytes" if unit == "bytes" else f"{size:.1f} {unit}"


def _fit(budget: float | None, cost: dict[int, float] | None, show) -> str:
    if budget is None:
        return ""
    if cost is None:
        return f"{show(budget)}, which cannot be checked until the cost is known"
    verdicts = [f"{_runs(runs)}: {'fits' if value <= budget else 'goes over'} ({show(value)})"
                for runs, value in cost.items()]
    return f"{show(budget)}; " + ", ".join(verdicts)


def _refusals(refused: list[tuple[Arm, str]]) -> list[str]:
    """One line for each reason within a model and mode, then the arms it skips. A refusal names the arm's own profile,
    which tells a person nothing its label does not, so that name is left out of the grouping."""
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for arm, why in refused:
        reason = why.replace(f"profile {arm.profile_id!r}", "the arm's profile")
        grouped.setdefault((arm.model, arm.mode, reason), []).append(arm.label)
    lines = []
    for (model, mode, reason), labels in grouped.items():
        lines += [f"  {model}, {mode}: {reason}", *(f"    {label}" for label in labels)]
    return lines


def render(cfg: SweepConfig, *, tokens: dict[str, tuple[float, float]] | None = None) -> str:
    """What `plan` prints, for a person deciding whether to run the sweep."""
    est = estimate(cfg, tokens=tokens)
    check = cfg.search.strategy == "check"
    (tuning, tuning_answerable), (held, held_answerable) = est.parts["tuning"], est.parts["held_out"]
    lines = [f"Sweep {cfg.sweep.name!r}: models {', '.join(cfg.models)}; modes {', '.join(cfg.modes)}; "
             f"strategy {cfg.search.strategy}"]
    if check:
        lines.append(f"Questions: {tuning} ({tuning_answerable} to answer), every one asked of each arm, with no split")
    else:
        lines.append(f"Questions: {tuning} to tune on ({tuning_answerable} to answer), {held} held out "
                     f"({held_answerable} to answer), split seed {cfg.dataset.split_seed}")

    lines += ["", "Rounds"]
    for (model, mode), rounds in est.rounds.items():
        lines.append(f"  {model}, {mode}")
        if not any(rounds.values()):
            lines.append("    nothing runs: the engine refuses the start (below)")
            continue
        for name, arms in rounds.items():
            labels = ", ".join(arm.label for arm in arms)
            if name == "grid" and cfg.search.strategy == "screen-then-grid":
                lines.append(f"    grid: at most {len(arms)} arms, if the screen keeps every value it tries" if arms
                             else "    grid: none, since no two settings vary together")
            elif name == "check":
                lines.append(f"    check: {labels}, on every question")
            else:
                lines.append(f"    {name}: {len(arms)} arms: {labels}")
        if not check and _searches(rounds):
            lines.append(f"    timing: the start and each finalist, once each and alone (the cost counts "
                         f"{TIMING_ARMS_PER_MODE}: the start and one finalist)")
    if not check:
        for model in cfg.models:
            if any(_searches(est.rounds[(model, mode)]) for mode in cfg.modes):
                lines.append(f"  confirm, {model}: the assembled profile against the start on the held-out part, in "
                             f"each mode")
    if est.refused:
        lines += ["", "Skipped, since the engine refuses them", *_refusals(est.refused)]

    lines += ["", "Cost"]
    asked = ", ".join(f"{count} with {_runs(runs)}" for runs, count in est.case_runs.items())
    once = "; the timing round runs once either way" if len(est.case_runs) > 1 and not check else ""
    lines.append(f"  questions asked: {asked}{once}")
    if est.hours is None:
        lines.append(f"  hours: {_UNKNOWN} ([search] seconds_per_question gives an estimate before then)")
    else:
        hours = ", ".join(f"{value:.1f} with {_runs(runs)}" for runs, value in est.hours.items())
        lines.append(f"  hours: {hours}, at {cfg.search.seconds_per_question:g} s a question and "
                     f"{cfg.machine.parallel} arm(s) at a time")
    if est.dollars is not None:
        dollars = ", ".join(f"{value:.2f} with {_runs(runs)}" for runs, value in est.dollars.items())
        lines.append(f"  dollars: {dollars}")
    elif tokens is None:
        lines.append(f"  dollars: {_UNKNOWN}")
    else:
        unpriced = [m for m in cfg.models if not _priced(cfg, m, tokens)]
        lines.append(f"  dollars: unknown, with no price or no measured tokens for {', '.join(unpriced)}")
    budget = [text for text in (_fit(cfg.search.budget_hours, est.hours, lambda v: f"{v:.1f} hours"),
                                _fit(cfg.search.budget_dollars, est.dollars, lambda v: f"{v:.2f} dollars")) if text]
    lines.append("  budget: " + ("; ".join(budget) if budget else "none set"))

    lines += ["", f"What the search can show (exact McNemar test, Holm's correction at alpha {cfg.propose.alpha:g})"]
    if check:
        lines.append("  check mode: the start against the engine's defaults, for harm; no search, so no gain to detect")
    elif not est.detectable:
        lines.append("  nothing is compared with the start: the space has no value to try")
    for (model, mode), gain in est.detectable.items():
        compared = est.compared[(model, mode)]
        if gain is None:
            lines.append(f"  {model}, {mode}: too small to search. With {compared} arms compared with the start, a win "
                         f"takes more than the {est.room_bound} answerable questions there are to tune on. Use check "
                         "mode ([search] strategy = \"check\"): the start against the engine's defaults on every "
                         "question, for harm.")
        else:
            lines.append(f"  {model}, {mode}: an arm needs at least {gain} more right with none lost, with {compared} "
                         f"arms compared with the start (the grid adds more). Room: at most {est.room_bound}, the "
                         "answerable questions to tune on; the start's misses decide once it has run.")

    lines += ["", "Store"]
    for store, size in est.store_bytes.items():
        if size is None:
            lines.append(f"  store copy: each arm running copies {store}, which cannot be read now, so its size is "
                         "unknown")
        else:
            lines.append(f"  store copy: {_size(size)} for each arm running, of {store}")
    if est.unpinned_sources:
        lines += ["", "Sources"]
        lines += [f"  {source}: source not pinned. A live database cannot be hashed, so a change to it during the "
                  "sweep would go unseen." for source in est.unpinned_sources]
    return "\n".join(lines) + "\n"
