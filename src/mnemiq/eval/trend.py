from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from dataclasses import asdict, dataclass

from mnemiq.eval.report import Report

_COLS = ("source_id", "run_at", "accuracy", "strict_accuracy", "total", "correct",
         "correct_facts", "wrong", "deferred_wrongly", "error")

_DDL = (
    "CREATE TABLE IF NOT EXISTS mnemiq_eval_run ("
    "source_id TEXT, run_at TEXT, accuracy DOUBLE PRECISION, "
    "strict_accuracy DOUBLE PRECISION, total INT, correct INT, "
    "correct_facts INT, wrong INT, deferred_wrongly INT, error INT)"
)


def _at(row: dict) -> datetime:
    """A row's `run_at` as a comparable instant, tolerant of one that will not parse.

    An unparseable stamp sorts to the beginning rather than raising: this runs inside the gate's
    baseline lookup, and a malformed row in the history is not a reason to refuse to compare -- but
    it is emphatically not a reason to let that row BE the baseline either.
    """
    try:
        parsed = datetime.fromisoformat(row["run_at"])
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class TrendUnavailable(RuntimeError):
    """The trend store could not be READ. Distinct from "no baseline yet", and the distinction
    is the finding: `last_run` swallowed every exception into the same `None` that means a
    first run, so a misconfigured control DSN disarmed the gate exactly like a missing file --
    and the Postgres path is the one that looks configured. One is an outage an operator must
    fix, the other is Tuesday. M2 and M6 removed this same collapse elsewhere (M34).
    """


@dataclass
class RunRecord:
    source_id: str
    run_at: str
    accuracy: float
    strict_accuracy: float
    total: int
    correct: int
    correct_facts: int
    wrong: int
    deferred_wrongly: int
    error: int


def _to_record(source_id: str, run_at: str, report: Report) -> RunRecord:
    return RunRecord(
        source_id=source_id, run_at=run_at, accuracy=report.accuracy,
        strict_accuracy=report.strict_accuracy, total=report.total, correct=report.correct,
        correct_facts=report.correct_facts, wrong=report.wrong,
        deferred_wrongly=report.deferred_wrongly, error=report.error,
    )


def record_run(control_dsn: str | None, source_id: str, report: Report,
               path: str | None = None, run_at: str = "") -> None:
    """Append a run to the trend. Postgres when a control DSN is set, else a local JSON.
    Fail-soft: a trend-write error never fails the eval run.

    `run_at` defaults to NOW, not to the empty string it used to. Every production caller
    omits it, so Postgres held rows whose ordering key was identical and `last_run`'s
    `ORDER BY run_at DESC LIMIT 1` returned an arbitrary historical row -- the gate comparing
    against whichever one the planner happened to pick. The JSON backend was unaffected only
    because it takes `rows[-1]` and never looks at the field.
    """
    rec = _to_record(source_id, run_at or datetime.now(UTC).isoformat(), report)
    try:
        if control_dsn:
            import psycopg

            with psycopg.connect(control_dsn, autocommit=True) as con:
                con.execute(_DDL)
                con.execute(
                    f"INSERT INTO mnemiq_eval_run ({', '.join(_COLS)}) "
                    f"VALUES ({', '.join(['%s'] * len(_COLS))})",
                    tuple(asdict(rec)[c] for c in _COLS),
                )
            return
        rows = []
        if path and os.path.exists(path):
            with open(path) as fh:
                rows = json.load(fh)
        rows.append(asdict(rec))
        if path:
            with open(path, "w") as fh:
                json.dump(rows, fh)
    except Exception:
        pass  # fail-soft


def last_run(control_dsn: str | None, source_id: str,
             path: str | None = None) -> RunRecord | None:
    try:
        if control_dsn:
            import psycopg

            with psycopg.connect(control_dsn, autocommit=True) as con:
                con.execute(_DDL)
                row = con.execute(
                    f"SELECT {', '.join(_COLS)} FROM mnemiq_eval_run WHERE source_id = %s "
                    "ORDER BY run_at DESC LIMIT 1",
                    (source_id,),
                ).fetchone()
                return RunRecord(**dict(zip(_COLS, row, strict=True))) if row else None
        if path and os.path.exists(path):
            with open(path) as fh:
                rows = [r for r in json.load(fh) if r["source_id"] == source_id]
            if not rows:
                return None
            # By `run_at`, matching what the Postgres backend does. This took `rows[-1]` -- the
            # last LINE -- so the two stores answered the same question differently the moment a
            # file was not in chronological order, and a trend file is edited by hand whenever a
            # level is accepted. It became load-bearing when a superseded 100.0 sample was kept
            # above the 96.0 that replaced it: any reordering silently restored the higher number
            # as the bar, with nightly reds on one unstable case as the only symptom.
            #
            # Undated rows are IGNORED while any dated row exists, and position decides only when
            # none is dated at all. They pre-date `run_at` defaulting to now and they still exist,
            # so an empty string must neither win by sorting nor win by sitting last.
            dated = [r for r in rows if r.get("run_at")]
            if not dated:
                return RunRecord(**rows[-1])
            # Parsed, not compared as text. ISO-8601 strings only sort correctly when every stamp
            # carries the same offset, and nothing enforces that: a row written `+00:00` and one
            # written `Z`, or any local offset, would order by their punctuation. Ties keep the
            # LAST row, which is what position-based selection did.
            return RunRecord(**max(enumerate(dated), key=lambda pair: (_at(pair[1]), pair[0]))[1])
    except Exception as exc:
        # Raise rather than return None. Returning None here is how the gate came to pass
        # forever: it is the same value that means "first run", so a broken store read as a
        # clean slate and `--gate` waved every run through.
        raise TrendUnavailable(str(exc)) from exc
    return None


def check_regression(report: Report, previous: RunRecord | None,
                     tolerance: float = 0.02) -> str | None:
    """A message if either accuracy dropped more than tolerance below the previous run, else None.

    **`tolerance` is a proxy for zero, not a tuned band.** Read it before changing it.

    Accuracy is got-the-facts over ANSWERABLE cases, and ACME has 25 of those, so the metric
    moves in steps of 1/25 = **4 percentage points**. A 2% tolerance is finer than one case,
    which means it does not permit a small drift -- it permits nothing. The gate fires the
    moment a single answerable case regresses, and every value below 0.04 behaves identically.

    So there is no threshold to calibrate here, only a choice between two behaviours: anything
    under 0.04 is "no case may regress", and 0.04-0.079 is "one case may regress". On 25 cases
    the second is most of the drift worth catching, so **widening this is not the right answer
    to a flapping gate** -- find out which case moved instead.

    The likeliest source of a flap is not the model: `run_acme` re-enriches every table in the source every
    run, so two runs compare engines over independently regenerated snapshots. If the nightly
    reddens on a night nothing changed, pin the snapshot rather than loosen this.

    Not calibrated against a deliberate same-config control, but not unevidenced either: two
    independent runs (2026-08-11 and 2026-08-12) produced identical counts -- 21/3/1/5 -- and
    failed on the same case. The nightly is now the standing variance experiment, at no cost.
    """
    if previous is None:
        return None
    # **Register M87.** Both rates, because `strict_accuracy` was computed, stored on every trend
    # row, and compared by nothing -- so half of a recorded 96.0/84.0 bar was advertised and
    # unguarded. The degradation it misses is specific and real: `CORRECT_FACTS` is "right data,
    # different shape" and is deliberately not a failure for the product metric, which is exactly
    # why the exact-match rate needs its own watch. Three exact matches decaying into differently
    # shaped ones leaves accuracy untouched and takes strict 84.0 -> 72.0, green throughout.
    #
    # Same tolerance, for the same reason: strict is also over `answerable`, so it moves in steps
    # of one case and anything under 0.04 means "no case may regress".
    #
    # It costs no extra flapping on the corpus this gates, measured rather than hoped: across the
    # twelve CI runs that produced a comparison the two rates moved in lockstep, twelve points
    # apart every time (88.0/76.0, 100.0/88.0, 96.0/84.0).
    for label, now, before in (
        ("got-the-facts accuracy", report.accuracy, previous.accuracy),
        ("exact-match accuracy", report.strict_accuracy, previous.strict_accuracy),
    ):
        if now < before - tolerance:
            # Named, because two numbers can fail and an operator told only "accuracy regressed"
            # goes looking at the wrong one.
            return (f"{label} regressed: {now:.1%} < previous {before:.1%} - "
                    f"{tolerance:.0%} tolerance")
    return None


def gate_outcome(report: Report, previous: RunRecord | None,
                 store_error: str | None = None) -> str | None:
    """The reason `--gate` should fail, or None to pass.

    `check_regression` answers one question -- did either rate drop? -- and answering "no" when
    there is nothing to compare is correct at that level. The GATE's question is different:
    *did this run get checked?* Reading an absent comparison as a pass is what made the
    mechanism unfailable, and it held while a published 100% drifted sixteen points with CI
    green throughout (M34).

    So a gate that could not compare FAILS, and says which of the two reasons applies. That is
    the same default `writes_enabled` took in M3: forgetting must fail closed, and a check
    that silently does nothing is the thing this project defines itself against.
    """
    if store_error is not None:
        return (f"the accuracy trend could not be read ({store_error}), so this run was not "
                "compared against anything")
    if previous is None:
        return ("no baseline is recorded for this source, so this run was not compared against "
                "anything -- record one with `mnemiq eval --record`, then gate against it")
    return check_regression(report, previous)


def should_record(record: bool, gate: bool, gate_failed: bool) -> bool:
    """Whether this run belongs in the trend.

    `--gate` used to imply recording, and it recorded BEFORE comparing. Armed, that is a
    ratchet rather than a gate: a regression fails one build, becomes the baseline, and every
    run after it passes against the lower number. One red build and then green forever, which
    is indistinguishable from the sixteen-point drift that was actually observed (M34).

    So a gate is a reader. Writing is `--record`'s job, and when both are asked for, a run
    that failed the gate is not written -- a number we just rejected must not become the one
    we measure against.
    """
    if not record:
        return False
    return not (gate and gate_failed)
