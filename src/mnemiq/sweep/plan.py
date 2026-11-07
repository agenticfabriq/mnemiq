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
from mnemiq.runtime import _resolve_verify_level
from mnemiq.sweep.arms import Arm, baselines, check_arms, grid_arms, refusal_under, screen_arms, start_settings
from mnemiq.sweep.config import _FILE_SOURCES, SweepConfig, _source_specs, load_questions
from mnemiq.sweep.split import split
from mnemiq.sweep.stats import detectable_gain

# The timing round runs each finalist once, alone, on the first `TIMING_CASES` cases to tune on. How many finalists
# there are is known only after the grid; `plan` counts the start and one finalist for each model and mode, and says so.
TIMING_ARMS_PER_MODE = 2
TIMING_CASES = 20
_UNKNOWN = "unknown until the first questions run"
_JUDGE_UNPRICED = "the judge's price is not known"
# The rounds that compare arms on the tuning part, by strategy; timing and confirm follow a search.
_ROUNDS_OF = {"screen-then-grid": ("screen", "grid"), "grid": ("grid",), "check": ("check",)}


@dataclass(frozen=True)
class Estimate:
    """What `render` says, as numbers.

    `arms` counts each round's arms that run, refusals left out; the confirm round counts the assembled profile and
    the start once in each mode, and timing counts `TIMING_ARMS_PER_MODE`. `case_runs`, `hours` and `dollars` are
    keyed by how many times each arm runs: one and two when the first round decides, the configured count otherwise.
    `detectable` and `compared` are per model and mode, for a sweep that searches. `judged_elsewhere` names the models
    whose judge runs on another endpoint or model, which leaves `dollars` unknown.
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
    judged_elsewhere: list[str]


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
    if _searches(cfg, rounds):
        alone += TIMING_ARMS_PER_MODE * min(TIMING_CASES, tuning)
    return parallel, alone


def _searches(cfg: SweepConfig, rounds: dict[str, list[Arm]]) -> bool:
    """Whether these rounds search, so that timing and confirm follow. Check mode never does: its defaults arm has
    changes whenever the start profile sets anything, and it is compared for harm, not searched."""
    return cfg.search.strategy != "check" and any(arm.changes for arms in rounds.values() for arm in arms)


def _judged_elsewhere(cfg: SweepConfig, model: str, rounds: list[dict[str, list[Arm]]],
                      deployments: list[tuple[str | None, Settings]]) -> bool:
    """Whether an arm of `model` that runs calls a judge on another endpoint or model than its writer's. A run measures
    tokens per question, not per endpoint, so the judge's tokens would be priced at the writer's rates. Which modes call
    the judge is the engine's rule: those whose verify level, after the deployment's override, is "full"."""
    start = start_settings(cfg, model)
    writer = (start["llm_base_url"], start["llm_model"])
    if (start["verify_base_url"] or writer[0], start["verify_model"] or writer[1]) == writer:
        return False
    overrides = {settings.verify_override for _, settings in deployments}
    for arm in (arm for own in rounds for arms in own.values() for arm in arms):
        values = start | dict(arm.changes)
        if any(_resolve_verify_level(values[f"{mode}.verify"], override) == "full"
               for mode in arm.modes for override in overrides):
            return True
    return False


def estimate(cfg: SweepConfig, *, tokens: dict[str, tuple[float, float]] | None = None) -> Estimate:
    """The numbers `render` shows. `tokens` is what a run measured, per model: the mean tokens in and out of one
    question."""
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
        confirming = [mode for mode, r in own.items() if _searches(cfg, r)]
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
    judged_elsewhere = [m for m in cfg.models
                        if _judged_elsewhere(cfg, m, [rounds[(m, mode)] for mode in cfg.modes], deployments)]
    dollars = None
    if tokens is not None and not judged_elsewhere and all(_priced(cfg, m, tokens) for m in cfg.models):
        dollars = {runs: sum(sum(by_model[m][runs]) * _per_question(cfg, m, tokens) for m in cfg.models)
                   for runs in runs_each}

    arms = Counter()
    for (model, mode), r in rounds.items():
        for name, kept in r.items():
            arms[name] += len(kept)
        if _searches(cfg, r):
            arms["timing"] += TIMING_ARMS_PER_MODE
    for model in cfg.models:
        if any(_searches(cfg, rounds[(model, mode)]) for mode in cfg.modes):
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
    for _, settings in deployments:
        store = os.path.abspath(settings.store_path)
        store_bytes[store] = os.path.getsize(store) if os.path.isfile(store) else None
    unpinned = unpinned_sources(deployments)

    return Estimate(
        arms=dict(arms), case_runs=case_runs, hours=hours, dollars=dollars, detectable=detectable, compared=compared,
        room_bound=room_bound, store_bytes=store_bytes, unpinned_sources=unpinned, refused=refused, rounds=rounds,
        parts={name: (len(part), sum(case.answerable for case in part)) for name, part in parts.items()},
        judged_elsewhere=judged_elsewhere)


def unpinned_sources(deployments: list[tuple[str | None, Settings]]) -> list[str]:
    """Each source of `deployments` (from `baselines`) that is not a file, and so cannot be pinned: by its id and kind,
    after its deployment's db_id when there are several. Never by its target, which can hold a credential."""
    unpinned = []
    for db_id, settings in deployments:
        for spec in _source_specs(settings):
            if spec.kind not in _FILE_SOURCES:
                unpinned.append(f"{spec.id} ({spec.kind})" if db_id is None else f"{db_id}: {spec.id} ({spec.kind})")
    return unpinned


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


def _fit(budget: float | None, cost: dict[int, float] | None, show, unknown: str = "the cost is not known yet") -> str:
    if budget is None:
        return ""
    if cost is None:
        return f"{show(budget)}, which cannot be checked: {unknown}"
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


def _nothing_compared(est: Estimate, key: tuple[str, str]) -> str:
    """Why no arm of one model and mode is compared with the start, for a sweep that searches."""
    if not any(est.rounds[key].values()):
        return "the engine refuses the start (see skipped)"
    if any((arm.model, arm.mode) == key for arm, _ in est.refused):
        return "the engine refuses every arm (see skipped)"
    return "the space has no value to try"


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
        if _searches(cfg, rounds):
            lines.append(f"    timing: the start and each finalist, once each and alone, on the first {TIMING_CASES} "
                         f"cases to tune on (the cost counts {TIMING_ARMS_PER_MODE}: the start and one finalist)")
    for model in cfg.models:
        if any(_searches(cfg, est.rounds[(model, mode)]) for mode in cfg.modes):
            lines.append(f"  confirm, {model}: the assembled profile against the start on the held-out part, in each "
                         "mode")
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
    elif est.judged_elsewhere:
        lines.append(f"  dollars: unknown: {_JUDGE_UNPRICED} ({', '.join(est.judged_elsewhere)} judged on another "
                     "endpoint or model, and a run's tokens cannot be split between writer and judge)")
    elif tokens is None:
        lines.append(f"  dollars: {_UNKNOWN}")
    else:
        unpriced = [m for m in cfg.models if not _priced(cfg, m, tokens)]
        lines.append(f"  dollars: unknown, with no price or no measured tokens for {', '.join(unpriced)}")
    budget = [text for text in (_fit(cfg.search.budget_hours, est.hours, lambda v: f"{v:.1f} hours"),
                                _fit(cfg.search.budget_dollars, est.dollars, lambda v: f"{v:.2f} dollars",
                                     _JUDGE_UNPRICED if est.judged_elsewhere else "the cost is not known yet"))
              if text]
    lines.append("  budget: " + ("; ".join(budget) if budget else "none set"))

    lines += ["", f"What the search can show (exact McNemar test, Holm's correction at alpha {cfg.propose.alpha:g})"]
    if check:
        lines.append("  check mode: the start against the engine's defaults, for harm; no search, so no gain to detect")
    elif not est.detectable:
        reasons = {key: _nothing_compared(est, key) for key in est.rounds}
        if len(set(reasons.values())) == 1:
            lines.append(f"  nothing is compared with the start: {next(iter(reasons.values()))}")
        else:
            lines += [f"  {model}, {mode}: nothing is compared with the start: {why}"
                      for (model, mode), why in reasons.items()]
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
