"""One arm of a sweep, run on the product path: its own copy of each store, its own profiles file, `build_runtime`, and
`Runtime.ask` as the sweep's identity, graded by `run_case`.

`run_arm` is what a worker process runs. It is a module-level function whose arguments pickle, and it reads nothing a
parent process set up, so it can run in a process of its own under the spawn start method. Each case and mode it answers
is one line of `rows.jsonl` in the arm's folder, written as soon as it is graded, so a stopped arm resumes where it
stopped.

An endpoint that fails 3 questions in a row is taken to have stopped answering: the arm stops, leaves those 3 out of its
rows, and lists them in the sweep's `retries.json`, so the next run asks them again. A case listed there that fails
again keeps its error. Only a model endpoint's failure counts toward the stop: the writer's or the judge's, read from
the engine's `AgentAnswer.model_error`, or a provider error raised out of `Runtime.ask` (the embedder's), read by the
client's own mapping. Never from an answer's words: one question's error (a prompt too long, SQL the source rejects, a
statement the source cancelled) is a counted error of the arm and never stops it. Each error row says who failed it, in
`error_kind`: the endpoint, the source, or the arm's own answer.

An arm's first run records the pinned files in `pins.json`, and every run of it, a resume included, checks its store
copies and the files it ends with against that record. A change makes the arm invalid for good: `invalid.json` is
final. A run of an arm holds its folder's lock, so two processes never write one arm's rows.

`run` drives the rounds, each arm in a worker process of its own started by the spawn method, so no worker inherits the
parent's runtime, and every arm once before any arm twice. The first round (the screen, the check round, or a grid with
no screen before it) decides two things. Under `repeats = "auto"` it runs the start twice among its other arms, and
every arm then runs once when the two runs wrote the same SQL for every case, else twice. And after the start's first
run, the search stops for that model and mode when the start missed fewer questions than an arm must gain to win.
Before each arm the budget is checked, so the sweep stops before an arm rather than during one. An endpoint that stops
answering stops the sweep once the arms running have finished. The sweep's own record is `sweep.json`, and its split
`split.json`, both in the out folder.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from urllib.parse import unquote, unquote_plus

from openai import APIError

from mnemiq.agent.loop import AgentAnswer
from mnemiq import redact
from mnemiq.config import _KEYWORD_PASSWORD, Settings, _split_userinfo, names_a_credential, without_credentials
from mnemiq.contract import DeferralReason, EvaluationCase, IdentityContext
from mnemiq.eval.ablation import names_table
from mnemiq.eval.harness import CaseResult, Outcome, run_case
from mnemiq.llm.client import ModelUnavailable, model_error_kind
from mnemiq.observability.answer_log import engine_commit
from mnemiq.runtime import Runtime, _authz, build_runtime, load_current_snapshot
from mnemiq.store.bootstrap import init_store
from mnemiq.sweep.arms import Arm, arm_of, baseline, baselines, grid_arms, profile_text, refusal_under, screen_arms
from mnemiq.sweep.config import (
    RETRIES, SMOKE, SPLIT, SWEEP, SweepConfig, SweepConfigError, _deployment_ids, _source_specs, deployment_settings,
    fingerprint, load_questions, pinned_files,
)
from mnemiq.sweep.plan import _ROUNDS_OF, _judged_elsewhere, _refusals, _rounds
from mnemiq.sweep.split import PARTS, saved_split, split, split_digest
from mnemiq.sweep.stats import detectable_gain

ROWS = "rows.jsonl"
PROFILE = "profile.toml"
INVALID = "invalid.json"
PINS = "pins.json"
# Each held by one process at a time: an arm's run folder by the worker running it, the out folder by one `run`.
LOCK = ".lock"
OUT_LOCK = ".sweep.lock"
# What a smoke writes in the folder it makes, so it never replaces a folder it did not make.
SMOKE_MARK = ".mnemiq-smoke"
# The rounds that follow a search, which `run` leaves to their own code.
_AFTER_SEARCH = ("timing", "confirm")
# Endpoint errors in a row that stop an arm.
_STREAK = 3
# Statuses that fail every question alike: a wrong or revoked key, a model name the server does not serve, a rate
# limit. Every 5xx joins them. Any other status is the one question's.
_ENDPOINT_STATUSES = frozenset({401, 403, 404, 429})
_STATUS = re.compile(r"status_(\d{3})")
# The one reason a failed answer gives when the source failed it: the source did not answer. Every other failed answer
# is the arm's own, `EXECUTION_FAILED` among them: the source rejected or timed out on every repair of the arm's SQL. A
# statement someone else cancelled reads the same way: a file source never cancels one, a live database can, and that
# case is accepted as the arm's.
_SOURCE_REASONS = frozenset({DeferralReason.SOURCE_UNAVAILABLE})


def engine_build(package: Path | None = None) -> dict:
    """The engine this process runs: its commit when that can be read (`engine_commit`), and `source_sha`.

    A wheel has no commit, and a checkout keeps one `-dirty` stamp through further edits, so the source hash is what
    tells two builds apart: a sha256 over each `.py` file of the package, in order of its path relative to the package,
    path and contents both.
    """
    if package is None:
        import mnemiq

        package = Path(mnemiq.__file__).resolve().parent
    files = sorted((path.relative_to(package).as_posix(), path) for path in package.rglob("*.py"))
    digest = hashlib.sha256()
    for relative, path in files:
        for part in (relative.encode(), path.read_bytes()):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return {"commit": engine_commit(package=package), "source_sha": digest.hexdigest()}


def _identity(cfg: SweepConfig) -> IdentityContext:
    who = cfg.dataset.identity
    return IdentityContext(tenant_id=who.tenant, principal_id=who.principal, roles=list(who.roles))


def _snapshot(settings: Settings):
    """The deployment's snapshot, read by the engine's own reader from a copy of its store: the reader opens a store to
    write, and the store is a pinned file."""
    with tempfile.TemporaryDirectory(prefix="mnemiq-sweep-store-") as folder:
        copy = os.path.join(folder, "store.duckdb")
        shutil.copyfile(settings.store_path, copy)
        con = init_store(copy, local_only=settings.local_only)
        try:
            snapshot, _ = load_current_snapshot(settings.model_copy(update={"store_path": copy, "control_dsn": None}),
                                                con)
        finally:
            con.close()
    return snapshot


def _unseen(snapshot, grants, cases: Sequence[EvaluationCase]) -> list[str]:
    """Each case whose gold reads a table the grants withhold, with those tables, read as `names_table` reads which
    tables a gold query is about."""
    withheld = sorted({column.object_id for column in snapshot.columns if not grants.allows(column.object_id)})
    found = []
    for case in cases:
        tables = [table for table in withheld if names_table(case.gold_sql or "", table)]
        if tables:
            found.append(f"{case.id} ({', '.join(tables)})")
    return found


def identity_refusal(cfg: SweepConfig) -> str | None:
    """Why the sweep's identity cannot be graded, or None. Checked for each deployment.

    An identity whose grants hold a row filter or a mask is refused: `run_case` runs the gold query unfiltered, while
    the answer's SQL carries the identity's filters and masks, so a right answer would be graded wrong. So are the cases
    whose gold reads a table the identity cannot see, listed by id.
    """
    identity = _identity(cfg)
    cases = [case for case in load_questions(cfg) if case.answerable]
    reasons = []
    for db_id in _deployment_ids(cfg):
        settings = deployment_settings(cfg, db_id)
        where = f"the deployment for {db_id!r}: " if db_id is not None else ""
        grants = _authz(settings).grants_for(identity)
        if not grants.available:
            reasons.append(f"{where}the access policy {settings.authz_path} cannot be read, so the identity is granted "
                           "nothing")
            continue
        if grants.row_filters or grants.pii_mask:
            reasons.append(f"{where}the identity's grants hold a row filter or a mask, and the gold queries run "
                           "without them, so a right answer would be graded wrong; sweep as an identity with neither")
            continue
        mine = [case for case in cases if cfg.dataset.deployments is None or case.db_id == db_id]
        if mine and (unseen := _unseen(_snapshot(settings), grants, mine)):
            reasons.append(f"{where}these cases' gold reads a table the identity cannot see, so no answer it can give "
                           f"is graded right: {', '.join(unseen)}")
    return "; ".join(reasons) or None


def arm_folder(cfg: SweepConfig, arm: Arm, repeat: int) -> Path:
    """Where one run of `arm` keeps its files: `<out>/<model>/<round>/<modes>/<profile_id>/run-<repeat>`, the modes
    joined by `+`. The start's `profile_id` is the same in every mode, so the mode is part of the path."""
    return Path(cfg.sweep.out) / arm.model / arm.round / "+".join(arm.modes) / arm.profile_id / f"run-{repeat}"


def _position(cfg: SweepConfig, db_id: str | None) -> int:
    return _deployment_ids(cfg).index(db_id)


def _store_copy(cfg: SweepConfig, folder: Path, db_id: str | None) -> Path:
    return Path(folder) / "stores" / f"store-{_position(cfg, db_id)}.duckdb"


def arm_settings(cfg: SweepConfig, arm: Arm, folder: Path, db_id: str | None) -> Settings:
    """The settings `arm` runs under against the deployment `db_id` reads: its `baseline` (every profile setting at the
    engine's field default, unset included, and no mode settings), with the arm's profiles file, its own copy of the
    store, its own answer log, and neither the shared result cache nor the trace sink."""
    settings = baseline(deployment_settings(cfg, db_id))
    folder = Path(folder)
    return settings.model_copy(update={
        "profiles_path": str(folder / PROFILE),
        "store_path": str(_store_copy(cfg, folder, db_id)),
        "answer_log_path": str(folder / f"answers-{_position(cfg, db_id)}.jsonl"),
        "control_dsn": None,
        "verity_traces_url": None,
        "mode_settings": None,
    })


def _sha256(path: Path | str) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _copy_stores(cfg: SweepConfig, folder: Path, db_ids: list[str | None], pinned: dict[str, str]) -> list[str]:
    """Copy each deployment's store into the arm's folder; return the stores whose copy is not the pinned file."""
    changed = []
    for db_id in db_ids:
        store = os.path.abspath(deployment_settings(cfg, db_id).store_path)
        copy = _store_copy(cfg, folder, db_id)
        copy.parent.mkdir(parents=True, exist_ok=True)
        # A log left by a runtime that stopped would be replayed onto the fresh copy.
        Path(f"{copy}.wal").unlink(missing_ok=True)
        shutil.copyfile(store, copy)
        if _sha256(copy) != pinned.get(store):
            changed.append(store)
    return changed


def _changed_since(cfg: SweepConfig, pinned: dict[str, str]) -> dict:
    """What moved among the pinned files since `pinned` was recorded; empty when nothing did."""
    try:
        now = pinned_files(cfg)
    except SweepConfigError as exc:
        return {"unreadable": str(exc)}
    changed = sorted(path for path in pinned.keys() | now.keys() if pinned.get(path) != now.get(path))
    return {"changed": changed} if changed else {}


def _replace(path: Path, text: str) -> None:
    """`path` written whole or not at all: a reader never sees half of it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value) -> None:
    _replace(path, json.dumps(value, indent=2, sort_keys=True))


def _read_rows(path: Path) -> list[dict]:
    """The rows written so far. A line cut short by a stopped process is dropped, and the file rewritten without it, so
    the next row starts on a line of its own."""
    if not path.exists():
        return []
    rows, torn = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            torn = True
    if torn:
        _replace(path, "".join(json.dumps(row) + "\n" for row in rows))
    return rows


def _rows_so_far(path: Path) -> list[dict]:
    """The rows written so far, read without changing the file: an arm may still be writing it, and a line it has not
    finished is left out."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _read_retries(cfg: SweepConfig) -> dict:
    """The sweep's `retries.json`: each arm run's folder, relative to the out folder, to the cases its endpoint stopped
    answering. It is replaced whole, never written in place, so it reads without the lock."""
    path = Path(cfg.sweep.out) / RETRIES
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@contextmanager
def _retries(cfg: SweepConfig) -> Iterator[dict]:
    """`retries.json` to change, held under a lock until the caller is done, then written back. Arms run in parallel
    processes, and a dead endpoint stops them together."""
    import fcntl  # POSIX: the lock is held by the process, and freed when it dies

    out = Path(cfg.sweep.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f".{RETRIES}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        retries = _read_retries(cfg)
        yield retries
        _write_json(out / RETRIES, retries)


class ArmInUse(RuntimeError):
    """Another process holds the run folder of an arm: a worker still running it."""


@contextmanager
def _sole(path: Path, refusal: Exception) -> Iterator[None]:
    """`path`'s lock, held until the caller is done; `refusal` is raised at once when another process holds it. The lock
    is the process's, and is freed when it dies."""
    import fcntl  # POSIX, as the retries lock

    with open(path, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise refusal from None
        yield


def _retry_key(cfg: SweepConfig, folder: Path) -> str:
    return Path(os.path.relpath(folder, cfg.sweep.out)).as_posix()


def _endpoint(model_error: str) -> bool:
    if model_error == "connect":
        return True
    status = _STATUS.fullmatch(model_error)
    return status is not None and (int(status[1]) in _ENDPOINT_STATUSES or 500 <= int(status[1]) <= 599)


def _error_kind(result: CaseResult, answer: AgentAnswer | None, model_error: str | None) -> str | None:
    """Who failed a row that is an error; None for a row that is not.

    `endpoint`: a model's server stopped answering, in a way that fails every question. `source`: the source failed us,
    not answering the arm's statement, or failing the gold query or the answer's re-run while it was graded.
    `answer`: the arm's own failure, as one question's model error, the engine raising, or SQL the source rejected or
    timed out on at every repair.
    """
    if result.outcome != Outcome.ERROR:
        return None
    if model_error is not None:
        return "endpoint" if _endpoint(model_error) else "answer"
    if answer is None:
        return "answer"
    if answer.failed and answer.reason_code not in _SOURCE_REASONS:
        return "answer"
    return "source"


def _ask(runtime: Runtime, case: EvaluationCase, mode: str, arm: Arm, identity: IdentityContext,
         build: dict) -> dict:
    """One case in one mode, answered and graded, as its row."""
    answers: list[AgentAnswer] = []
    raised: list[str] = []  # the kind of a model error `Runtime.ask` raised rather than answered
    clock: dict[str, float] = {}

    def emit(event: dict) -> None:
        if event.get("kind") == "result":
            clock.setdefault("rows", time.perf_counter())

    def engine(question: str) -> AgentAnswer:
        clock["asked"] = time.perf_counter()
        try:
            answer = runtime.ask(question, identity, mode=mode, emit=emit, profile=arm.profile_id)
        except APIError as exc:  # the provider's own error, from a model call no client wraps: the embedder's
            raised.append(model_error_kind(exc))
            raise
        except ModelUnavailable as exc:
            raised.append(exc.kind)
            raise
        answers.append(answer)
        return answer

    before = runtime.usage(arm.profile_id)
    result = run_case(case, engine, runtime.adapter)
    after = runtime.usage(arm.profile_id)
    answer = answers[-1] if answers else None
    model_error = answer.model_error if answer is not None else (raised[-1] if raised else None)
    return {
        "id": case.id,
        "mode": mode,
        "outcome": str(result.outcome),
        "made_up": not case.answerable and result.outcome == Outcome.WRONG,
        "sql": result.sql,
        "ms": result.ms,
        "rows_ms": (clock["rows"] - clock["asked"]) * 1000 if "rows" in clock else None,
        "calls": after.calls - before.calls,
        "prompt_tokens": after.prompt_tokens - before.prompt_tokens,
        "completion_tokens": after.completion_tokens - before.completion_tokens,
        "profile_version": runtime.profiles[arm.profile_id].version,
        "engine_build": build,
        "error_kind": _error_kind(result, answer, model_error),
        "model_error": model_error,
        "verify_layer": answer.verify_layer if answer is not None else None,
    }


def _answer_all(cfg: SweepConfig, arm: Arm, cases: Sequence[EvaluationCase], folder: Path,
                runtimes: dict[str | None, Runtime], build: dict, before: Callable[[str, str], None] | None) -> bool:
    """Ask every case and mode not yet in `rows.jsonl`, calling `before` with the case's id and the mode just before
    each is asked; True when the endpoint stopped answering."""
    identity = _identity(cfg)
    rows = folder / ROWS
    done = {(row.get("id"), row.get("mode")) for row in _read_rows(rows)}
    key = _retry_key(cfg, folder)
    retried = {(entry["id"], entry["mode"]) for entry in _read_retries(cfg).get(key, [])}
    held: list[dict] = []  # endpoint errors in a row, not yet written
    with open(rows, "a", encoding="utf-8") as out:
        def write(row: dict) -> None:
            out.write(json.dumps(row) + "\n")
            out.flush()

        for case in cases:
            runtime = runtimes[case.db_id if cfg.dataset.deployments is not None else None]
            for mode in arm.modes:
                if (case.id, mode) in done:
                    continue
                if before is not None:
                    before(case.id, mode)
                row = _ask(runtime, case, mode, arm, identity, build)
                if row["error_kind"] == "endpoint" and (case.id, mode) in retried:
                    write(row)  # failed again after a stop: kept as an error, and no part of a new streak
                    continue
                if row["error_kind"] == "endpoint":
                    held.append(row)
                    if len(held) == _STREAK:
                        with _retries(cfg) as retries:
                            retries.setdefault(key, []).extend(
                                {"id": error["id"], "mode": error["mode"], "model_error": error["model_error"]}
                                for error in held)
                        return True
                    continue
                for error in held:  # the streak ended: each was that question's own error
                    write(error)
                held.clear()
                write(row)
        for error in held:
            write(error)
    return False


def run_arm(cfg: SweepConfig, arm: Arm, cases: Sequence[EvaluationCase], folder: Path,
            before: Callable[[str, str], None] | None = None) -> str:
    """Run `arm` on `cases` into `folder`, resuming from the rows already there.

    An arm with an `invalid.json` is not run again. Its first run records the pinned files in `pins.json`; every run
    checks the files against that record, copies each store the cases read and checks each copy against it, writes the
    arm's `profile.toml`, and builds one runtime per deployment. Each case not yet answered is asked as the config's
    identity in each of the arm's modes, graded by `run_case`, and appended to `rows.jsonl` as one flushed row per case
    and mode, `before` (when given) called with the case's id and the mode just before it is asked. When the run ends,
    the pinned files are read again, and `invalid.json` records any that moved from the record.

    Returns `"invalid"` (a pinned file moved, in this run or before it), else `"endpoint"` (the endpoint stopped
    answering; the rows stand, and the next run resumes), else `"complete"`. It holds the folder's lock while it runs,
    and raises `ArmInUse` when another process holds it, such as a worker a stopped sweep left running.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    busy = ArmInUse(f"another process is running the arm in {folder}, as a worker a stopped sweep left running would; "
                    "run again once it has ended")
    with _sole(folder / LOCK, busy):
        return _run_arm(cfg, arm, cases, folder, before)


def _run_arm(cfg: SweepConfig, arm: Arm, cases: Sequence[EvaluationCase], folder: Path,
             before: Callable[[str, str], None] | None) -> str:
    """`run_arm` itself, with the folder made and its lock held."""
    if (folder / INVALID).exists():
        return "invalid"
    if (folder / PINS).exists():
        pinned = json.loads((folder / PINS).read_text(encoding="utf-8"))
    else:
        pinned = pinned_files(cfg)
        _write_json(folder / PINS, pinned)
    if moved := _changed_since(cfg, pinned):
        _write_json(folder / INVALID, moved)
        return "invalid"
    build = engine_build()
    db_ids = [None] if cfg.dataset.deployments is None else sorted({case.db_id for case in cases})
    if changed := _copy_stores(cfg, folder, db_ids, pinned):
        _write_json(folder / INVALID, {"changed": changed})
        return "invalid"
    _replace(folder / PROFILE, profile_text(cfg, arm))
    runtimes: dict[str | None, Runtime] = {}
    try:
        for db_id in db_ids:
            runtimes[db_id] = build_runtime(arm_settings(cfg, arm, folder, db_id))
        stopped = _answer_all(cfg, arm, cases, folder, runtimes, build, before)
    finally:
        for runtime in runtimes.values():
            runtime.con.close()
    if moved := _changed_since(cfg, pinned):
        _write_json(folder / INVALID, moved)
        return "invalid"
    return "endpoint" if stopped else "complete"


# --- the rounds -------------------------------------------------------------------------------------------------------


def _say(*lines: str) -> None:
    """Lines for the person running the sweep, as it happens."""
    for line in lines:
        print(line, flush=True)


def _first_round(cfg: SweepConfig) -> str:
    """The round that decides the repeats and measures the room: the screen, the check round, or the grid when no screen
    runs before it."""
    return _ROUNDS_OF[cfg.search.strategy][0]


def _owed(arm: Arm, cases: Sequence[EvaluationCase]) -> set[tuple[str, str]]:
    return {(case.id, mode) for case in cases for mode in arm.modes}


def _left(folder: Path, owed: set[tuple[str, str]]) -> int:
    """How many of the cases and modes `owed` the run in `folder` has yet to answer. Nothing, once it is invalid: that
    is final, so an invalid run is reported and never run again, and its cases in `retries.json` are no work left."""
    if (folder / INVALID).exists():
        return 0
    return len(owed - {(row.get("id"), row.get("mode")) for row in _rows_so_far(folder / ROWS)})


def _parts(cfg: SweepConfig) -> dict[str, list[EvaluationCase]]:
    """The cases to tune on and those held out: the sweep's saved split, read back and verified (`saved_split`), once a
    run has started the sweep, else drawn by `split`."""
    saved = saved_split(cfg)
    return saved if saved is not None else split(cfg, load_questions(cfg))


def _rows_exist(out: Path) -> bool:
    """Whether an arm of a round has written rows in the out folder `out`, where `arm_folder` puts them. A smoke's sit a
    folder deeper, in the smoke folder, and are not the sweep's."""
    return any(out.glob(f"*/*/*/*/run-*/{ROWS}"))


# The stops a later run clears, as it resumes the sweep: for the budget, a dead endpoint, a dead worker.
_CLEARED = ("budget", "endpoint", "worker")


def _begin(cfg: SweepConfig) -> dict:
    """The sweep's record, `sweep.json`, beside its split, `split.json`. A first run writes both. A later run refuses a
    sweep started under another config (the fingerprint) or another engine build (its source hash), and rows with no
    record or no split to read them by, all before any arm runs. A stop for the budget, a dead endpoint or a dead worker
    is cleared, as this run resumes it. A new sweep draws its split, over any that another left, and records its
    digest; a sweep whose split is gone may draw it again only before any row, and a saved split that is not the one
    drawn is refused."""
    out = Path(cfg.sweep.out)
    record, parts = out / SWEEP, out / SPLIT
    found, build = fingerprint(cfg), engine_build()
    new = not record.exists()
    if not new:
        state = json.loads(record.read_text(encoding="utf-8"))
        if state.get("fingerprint") != found:
            raise SweepConfigError(
                f"{record}: this sweep was started under another config. The config, a deployment's settings, or a "
                "file they pin (the question file, a start profile, an access policy, a store, a file source) has "
                "changed since. Put it back, or start a new sweep in another out folder.")
        started = str((state.get("engine_build") or {}).get("source_sha"))
        if started != build["source_sha"]:
            raise SweepConfigError(
                f"{record}: this sweep was started under another engine build (source {started[:12]}, now "
                f"{build['source_sha'][:12]}), and its rounds must match on the engine's source. Run the build it "
                "started with, or start a new sweep in another out folder.")
        if parts.exists():
            saved_split(cfg)  # refuses a split that is not the one the sweep drew
    elif _rows_exist(out):
        raise SweepConfigError(f"{out} holds rows but no {SWEEP}, so the config they were written under cannot be "
                               "checked. Start a new sweep in another out folder.")
    else:
        state = {"fingerprint": found, "engine_build": build, "split_seed": cfg.dataset.split_seed}
        if isinstance(cfg.sweep.repeats, int):
            state["repeats"] = cfg.sweep.repeats
    if new or not parts.exists():
        if not new and _rows_exist(out):
            raise SweepConfigError(f"{out} holds rows but no {SPLIT}, so the cases they answered cannot be read back. "
                                   "Start a new sweep in another out folder.")
        out.mkdir(parents=True, exist_ok=True)
        drawn = split(cfg, load_questions(cfg))
        ids = {part: [case.id for case in drawn[part]] for part in PARTS}
        _write_json(parts, ids)
        state["split_digest"] = split_digest(ids)
    if state.get("stopped") in _CLEARED:
        del state["stopped"]
    _write_json(record, state)
    return state


def repeats_needed(cfg: SweepConfig) -> int:
    """How many runs each arm owes, as the start's two runs in the first round decide it: 1 when they wrote the same SQL
    for every case, else 2.

    Each model and mode whose start has finished both runs is read, and a difference in any decides 2: the server
    batches the arms' requests together, so a start that does not repeat says the same of the arms beside it. A run
    that is not valid shows nothing either way, so when no start has two valid runs to compare, the answer is 2 too.
    """
    tuning = _parts(cfg)["tuning"]
    finished = compared = 0
    for model in cfg.models:
        for mode in cfg.modes:
            start = arm_of(model, mode, {}, round=_first_round(cfg))
            folders = [arm_folder(cfg, start, repeat) for repeat in (1, 2)]
            if any(not folder.is_dir() or _left(folder, _owed(start, tuning)) for folder in folders):
                continue
            finished += 1
            if any((folder / INVALID).exists() for folder in folders):
                continue
            first, second = ({(row.get("id"), row.get("mode")): row.get("sql") for row in _rows_so_far(folder / ROWS)}
                             for folder in folders)
            if first != second:
                return 2
            compared += 1
    if not finished:
        raise ValueError("no start has finished two runs of the first round, so whether runs repeat is not known yet")
    return 1 if compared else 2


def room(cfg: SweepConfig, model: str, mode: str) -> int:
    """The most an arm of `model` in `mode` can gain: the answerable cases to tune on that its start did not get right
    in its first run of the first round, right as the report reads it."""
    from mnemiq.sweep.report import tally  # the report reads this module's files, so it is imported when needed

    first = _first_round(cfg)
    if first == "check":
        raise ValueError("check mode does not search, so it has no room to measure")
    answerable = [case for case in _parts(cfg)["tuning"] if case.answerable]
    start = arm_of(model, mode, {}, round=first)
    folder = arm_folder(cfg, start, 1)
    if (folder / INVALID).exists() or _left(folder, _owed(start, answerable)):
        raise ValueError(f"the start of {model} in {mode} has no first run that is valid and answered every case")
    ids = {case.id for case in answerable}
    return len(answerable) - tally([row for row in _rows_so_far(folder / ROWS) if row.get("id") in ids], mode).right


def _endpoints(cfg: SweepConfig, model: str) -> str:
    """Where the arms of `model` call a model, each URL without its credentials: its writer, its judge when the config
    names one, and each deployment's embedder."""
    entry = cfg.models[model]
    judge = cfg.judge if cfg.judge is not None else entry
    called = [("the writer", entry.llm_base_url), ("the judge", judge.verify_base_url)]
    called += [("the embedder", deployment_settings(cfg, db_id).embed_endpoint()[0]) for db_id in _deployment_ids(cfg)]
    by_url: dict[str, list[str]] = {}
    for role, url in called:
        if url:
            roles = by_url.setdefault(without_credentials(url), [])
            if role not in roles:
                roles.append(role)
    return "; ".join(f"{' and '.join(roles)} at {url}" for url, roles in by_url.items())


# A known secret shorter than this would match ordinary words; `redact.scrub_credentials` still hides one written as a
# password field or a URL's userinfo.
_SHORTEST_SECRET = 4
_WEB = ("http", "https")


def _credentials_in(locator: str) -> list[str]:
    """The secret parts of a DSN or URL, read as `without_credentials` reads them: the password before its host's `@`
    (the user too, for a web URL, whose user is part of its credential), each query value named like a credential, and
    a keyword DSN's password. Each as written and decoded."""
    found: list[str] = []
    prefix, userinfo, rest = _split_userinfo(locator)
    if userinfo:
        user, colon, password = userinfo.partition(":")
        if not colon:  # an Easy Connect string's user/password
            user, _, password = userinfo.partition("/")
        found += [password, unquote(password)]
        if prefix[:-3].lower() in _WEB:
            found += [user, unquote(user)]
    for pair in filter(None, rest.partition("#")[0].partition("?")[2].split("&")):
        name, _, value = pair.partition("=")
        if names_a_credential(unquote_plus(name)):
            found += [value, unquote_plus(value)]
    for field in _KEYWORD_PASSWORD.finditer(locator):
        value = field.group(0).split("=", 1)[1].strip()
        found += [value, value.strip("'")]
    return found


def _known_secrets(cfg: SweepConfig) -> list[tuple[str, str]]:
    """Each secret the sweep's settings and environment hold, longest first, with what a message shows in its place.

    A key, password or secret field of each deployment's settings, and the value of each key variable the config names,
    show as `***`. A DSN or URL field, and each source's target, shows without its credentials, and its password (its
    user too, for a web URL) as `***`. Read once, at the start of a run, so a failure never reads an env file again.
    """
    found: dict[str, str] = {}

    def hide(value: object) -> None:
        if isinstance(value, str) and len(value) >= _SHORTEST_SECRET:
            found[value] = "***"

    def locate(value: object) -> None:
        if not isinstance(value, str) or not value:
            return
        if (shown := without_credentials(value)) != value:
            found.setdefault(value, shown)
        for part in _credentials_in(value):
            hide(part)

    for db_id in _deployment_ids(cfg):
        settings = deployment_settings(cfg, db_id)
        for name in Settings.model_fields:
            if name.endswith(("_dsn", "_url")):
                locate(getattr(settings, name))
            elif any(hint in name for hint in ("key", "password", "secret")):
                hide(getattr(settings, name))
        for spec in _source_specs(settings):
            locate(spec.target)
    for entry in (*cfg.models.values(), *([cfg.judge] if cfg.judge is not None else [])):
        for variable in (getattr(entry, "api_key_env", None), entry.verify_api_key_env):
            if variable:
                hide(os.environ.get(variable))
    return sorted(found.items(), key=lambda pair: len(pair[0]), reverse=True)


def _hidden(text: str, secrets: Sequence[tuple[str, str]]) -> str:
    """`text` as a message may show it: each known secret replaced (`_known_secrets`), then every URL's userinfo and
    every password field, known or not (`redact.scrub_credentials`)."""
    for value, shown in secrets:
        text = text.replace(value, shown)
    return redact.scrub_credentials(text)


def _exception_line(exc: BaseException) -> str:
    """An exception's own `Type: message` line, as a traceback names it, on one line: none of its notes, and none of the
    causes chained to it."""
    kind = type(exc)
    name = kind.__qualname__
    if kind.__module__ not in ("builtins", "__main__"):
        name = f"{kind.__module__}.{name}"
    words = " ".join(str(exc).split())
    return f"{name}: {words}" if words else name


def _work(cfg: SweepConfig, arm: Arm, cases: list[EvaluationCase], folder: Path, conn,
          secrets: Sequence[tuple[str, str]], before: Callable[[str, str], None] | None = None) -> None:
    """What a worker process runs: one arm, with `before` called ahead of each case, then how it ended, sent back on
    `conn`.

    No exception leaves it: the process would print the traceback, chained causes and all, and building a runtime
    attaches each source with its DSN, which an error may quote. The exception's one line goes back instead, as
    `("raised", line)`, hidden by `_hidden` with `secrets`, and the process exits with status 1, printing nothing. A
    worker killed outright sends nothing.
    """
    try:
        try:
            outcome = run_arm(cfg, arm, cases, folder, before)
        except BaseException as exc:
            with suppress(BaseException):  # the parent may be gone, or a second interrupt may land mid-send
                conn.send(("raised", _hidden(_exception_line(exc), secrets)))
            raise SystemExit(1) from None  # an int status, for which a worker process prints nothing
        conn.send(outcome)
    finally:
        conn.close()


@dataclass(frozen=True)
class _Job:
    """One run of one arm: run `repeat` of `arm`, on `cases`, in `folder`, calling `before` ahead of each case (it is
    sent to the worker, so it must pickle)."""

    arm: Arm
    repeat: int
    cases: tuple[EvaluationCase, ...]
    folder: Path
    before: Callable[[str, str], None] | None = None

    def left(self) -> int:
        return _left(self.folder, _owed(self.arm, self.cases))

    def finished(self) -> bool:
        """Whether this run answered every case it owes, and is valid."""
        return not (self.folder / INVALID).exists() and not self.left()

    def name(self) -> str:
        return f"{self.arm.model}, {'+'.join(self.arm.modes)}, {self.arm.round}: {self.arm.label}, run {self.repeat}"


def _not_valid(job: _Job) -> str:
    return f"{job.name()}: not valid, since a file it pins moved (see {job.folder / INVALID}); it is not run again"


class _Meter:
    """What the sweep's rows show of its pace and its spend, per model: seconds and dollars a question. The rows already
    written are read when it starts, and each run's again when that run ends."""

    def __init__(self, cfg: SweepConfig) -> None:
        self.cfg = cfg
        self.runs: dict[Path, tuple[str, float, int, float, int]] = {}  # model, seconds, rows timed, dollars, rows
        for model in cfg.models:
            for rows in (Path(cfg.sweep.out) / model).glob(f"*/*/*/run-*/{ROWS}"):
                self.read(model, rows.parent)

    def read(self, model: str, folder: Path) -> None:
        entry = self.cfg.models[model]
        rows = _rows_so_far(folder / ROWS)
        timed = [row["ms"] for row in rows if isinstance(row.get("ms"), (int, float))]
        dollars = sum((row.get("prompt_tokens") or 0) * (entry.price_in_per_mtok or 0)
                      + (row.get("completion_tokens") or 0) * (entry.price_out_per_mtok or 0) for row in rows) / 1e6
        self.runs[folder] = (model, sum(timed) / 1000, len(timed), dollars, len(rows))

    def seconds_a_question(self, model: str) -> float | None:
        """Measured, or `[search] seconds_per_question` until a question of `model` has run."""
        mine = [run for run in self.runs.values() if run[0] == model]
        timed = sum(run[2] for run in mine)
        return sum(run[1] for run in mine) / timed if timed else self.cfg.search.seconds_per_question

    def dollars(self) -> float:
        return sum(run[3] for run in self.runs.values())

    def dollars_a_question(self, model: str) -> float | None:
        mine = [run for run in self.runs.values() if run[0] == model]
        rows = sum(run[4] for run in mine)
        return sum(run[3] for run in mine) / rows if rows else None


class _Rounds:
    """One `run` of a sweep, or of a smoke: the arms each round runs, the worker processes running them, what
    `sweep.json` records, and why the run stopped. A smoke has no `state`: it records nothing and has no budget, room or
    repeats."""

    def __init__(self, cfg: SweepConfig, state: dict | None, parts: dict[str, list[EvaluationCase]]) -> None:
        self.cfg = cfg
        self.state = state
        self.parts = parts
        self.deployments = baselines(cfg)
        self.first = _first_round(cfg)
        # The first round's arms are the ones `plan` counts, the start first, the arms the engine refuses left out.
        screen = [] if cfg.search.strategy == "check" else screen_arms(cfg)
        self.plans: dict[tuple[str, str], dict[str, list[Arm]]] = {}
        refused: list[tuple[Arm, str]] = []
        for model in cfg.models:
            for mode in cfg.modes:
                self.plans[(model, mode)], skipped = _rounds(cfg, model, mode, screen, self.deployments)
                refused += [(arm, why) for arm, why in skipped if arm.round == self.first]
        self.arms = {key: rounds[self.first] for key, rounds in self.plans.items()}
        # Check mode compares for harm and does not search, so it has no room to measure.
        self.compared = {key: sum(1 for arm in arms if arm.changes) if self.first != "check" else 0
                         for key, arms in self.arms.items()}
        if refused:
            _say("Skipped, since the engine refuses them:", *_refusals(refused))
        # Read once, here, so saying what failed never reads an env file again.
        self.secrets = _known_secrets(cfg)
        self.endpoints = {model: _endpoints(cfg, model) for model in cfg.models}
        self.clock = time.monotonic()
        self.seconds = state.get("seconds", 0.0) if state is not None else 0.0
        self.halt: str | None = None
        self.dead: list[str] = []  # the models with an endpoint that stopped answering
        self.died: list[str] = []  # the runs whose worker stopped without finishing
        self.meter = _Meter(cfg) if state is not None else None
        self.dollars_checked = state is not None and self._dollars_checkable()
        if state is not None:
            self._record_starts(refused)
            self._settle()

    # --- what the record says -----------------------------------------------------------------------------------

    def _no_room(self, key: tuple[str, str]) -> bool:
        model, mode = key
        return (self.state or {}).get("room", {}).get(model, {}).get(mode, {}).get("stopped") == "no room"

    def _record_starts(self, refused: Sequence[tuple[Arm, str]]) -> None:
        """Record in `starts`, for each model and mode, None when the engine runs its start, else why it refuses it. A
        mode recorded as running stays so: its rows are what the sweep ran, and a proposal reads this record rather
        than ask the engine again under another environment or build."""
        record = self.state.setdefault("starts", {})
        for (model, mode), arms in self.arms.items():
            known = record.setdefault(model, {})
            if arms:
                known[mode] = None
            elif known.get(mode, "") is not None:
                known[mode] = next((why for arm, why in refused if (arm.model, arm.mode, arm.profile_id)
                                    == (model, mode, "start")), "the engine refuses the start")

    def _active(self, key: tuple[str, str]) -> bool:
        """Whether `key`'s model and mode still runs arms: the engine runs its start, and it has not run out of room."""
        return bool(self.arms[key]) and not self._no_room(key)

    def _settle(self) -> None:
        """Record that the search stopped for want of room once every model and mode that searches has."""
        searched = [key for key in self.arms if self.arms[key] and self.compared[key]]
        if "stopped" not in self.state and searched and all(self._no_room(key) for key in searched):
            self.state["stopped"] = "no room"

    def elapsed(self) -> float:
        """Seconds this sweep's runs have taken, this one's so far included."""
        return self.seconds + time.monotonic() - self.clock

    def save(self) -> None:
        if self.state is not None:
            self.state["seconds"] = round(self.elapsed(), 3)
            _write_json(Path(self.cfg.sweep.out) / SWEEP, self.state)

    @property
    def halted(self) -> bool:
        return self.halt is not None

    def _stop(self, why: str) -> None:
        """Start no further arm, and record why; the next run clears it."""
        self.halt = self.halt or why
        if self.state is not None:
            self.state["stopped"] = why
            self.save()

    # --- the budget ---------------------------------------------------------------------------------------------

    def _dollars_checkable(self) -> bool:
        """Whether the dollar budget can be checked: every model has a price, and none is judged on another endpoint or
        model, whose tokens a row cannot tell from the writer's. Said once when it cannot."""
        if self.cfg.search.budget_dollars is None:
            return False
        unpriced = [model for model, entry in self.cfg.models.items()
                    if entry.price_in_per_mtok is None or entry.price_out_per_mtok is None]
        elsewhere = [model for model in self.cfg.models
                     if _judged_elsewhere(self.cfg, model, [self.plans[(model, mode)] for mode in self.cfg.modes],
                                          self.deployments)]
        if unpriced or elsewhere:
            why = [f"no price for {', '.join(unpriced)}"] if unpriced else []
            why += [f"{', '.join(elsewhere)} judged on another endpoint or model"] if elsewhere else []
            _say(f"The dollar budget cannot be checked ({'; '.join(why)}), so this run goes on without it.")
            return False
        return True

    def _over_budget(self, job: _Job, left: int, running: Sequence[tuple[_Job, int]]) -> str | None:
        """Why running `job`'s `left` questions would take the sweep past its budget, or None. With no pace measured
        yet, and none in `seconds_per_question`, the arm may start: only running one measures it.

        The runs going on beside it, each with the questions it had left when it started, run in the same hours, but
        each spends its own dollars, which its rows show only once it ends: so they are counted as committed."""
        search = self.cfg.search
        if search.budget_hours is not None:
            spare = search.budget_hours * 3600 - self.elapsed()
            pace = self.meter.seconds_a_question(job.arm.model)
            if spare <= 0:
                return f"the {search.budget_hours:g} hours budgeted are used"
            if pace is not None and pace * left > spare:
                return (f"it needs about {pace * left:.0f} s, and {spare:.0f} s of the {search.budget_hours:g} hours "
                        "budgeted are left")
        if self.dollars_checked:
            committed = sum((self.meter.dollars_a_question(other.arm.model) or 0.0) * asked
                            for other, asked in running)
            spare = search.budget_dollars - self.meter.dollars() - committed
            pace = self.meter.dollars_a_question(job.arm.model)
            if spare <= 0:
                return f"the {search.budget_dollars:g} dollars budgeted are spent"
            if pace is not None and pace * left > spare:
                return (f"it needs about {pace * left:.2f} dollars, and {spare:.2f} of the "
                        f"{search.budget_dollars:g} budgeted are left")
        return None

    # --- the workers --------------------------------------------------------------------------------------------

    def job(self, arm: Arm, repeat: int, cases: Sequence[EvaluationCase] | None = None,
            before: Callable[[str, str], None] | None = None) -> _Job:
        """Run `repeat` of `arm`, on `cases` (by default the cases to tune on)."""
        cases = self.parts["tuning"] if cases is None else cases
        return _Job(arm, repeat, tuple(cases), arm_folder(self.cfg, arm, repeat), before)

    def drain(self, pending: deque[_Job], advance: Callable[[deque[_Job]], None] | None = None, *,
              parallel: int | None = None) -> None:
        """Run the jobs in `pending` in turn, `parallel` at a time (by default `[machine] parallel`), each in a spawned
        worker of its own, until every one has run or the sweep stops. A run already done is skipped, and so is a run of
        the search for a model and mode that has run out of room. `advance` is called at the start and after each run
        ends, and may queue more. When the sweep stops, the runs going on finish and none starts."""
        limit = parallel or self.cfg.machine.parallel
        context = multiprocessing.get_context("spawn")
        running: dict[int, tuple[_Job, multiprocessing.process.BaseProcess, object, int]] = {}
        try:
            if advance is not None:
                advance(pending)
            while True:
                while pending and len(running) < limit and not self.halted:
                    job = pending.popleft()
                    left = job.left()
                    if not left and (job.folder / INVALID).exists():
                        _say(_not_valid(job))
                    searching = job.arm.round not in _AFTER_SEARCH
                    if not left or (self.state is not None and searching
                                    and not self._active((job.arm.model, job.arm.mode))):
                        continue
                    beside = [(other, asked) for other, _, _, asked in running.values()]
                    if self.state is not None and (why := self._over_budget(job, left, beside)):
                        _say(f"Stopped before {job.name()}: {why}. Raise the budget in [search] and run again; the "
                             "sweep resumes here.")
                        self._stop("budget")
                        break
                    receiver, sender = context.Pipe(duplex=False)
                    process = context.Process(target=_work, name=f"mnemiq-sweep-{job.arm.profile_id}",
                                              args=(self.cfg, job.arm, list(job.cases), job.folder, sender,
                                                    self.secrets, job.before))
                    process.start()
                    sender.close()
                    running[process.sentinel] = (job, process, receiver, left)
                if not running:
                    return
                for sentinel in wait(list(running)):
                    job, process, receiver, _ = running.pop(sentinel)
                    try:
                        outcome = receiver.recv() if receiver.poll() else None
                    except EOFError:
                        outcome = None
                    receiver.close()
                    process.join()
                    self._ended(job, outcome, process.exitcode)
                if advance is not None and not self.halted:
                    advance(pending)
        finally:
            # Reached with workers still running only when this process itself was stopped, as by Ctrl-C.
            for _, process, _, _ in running.values():
                process.terminate()
            for _, process, receiver, _ in running.values():
                process.join()
                receiver.close()
            self.save()

    def _ended(self, job: _Job, outcome: str | None, exitcode: int | None) -> None:
        if self.meter is not None:
            self.meter.read(job.arm.model, job.folder)
        if outcome == "complete":
            _say(f"{job.name()}: complete")
        elif outcome == "invalid":
            _say(_not_valid(job))
        elif outcome == "endpoint":
            _say(f"{job.name()}: stopped, since an endpoint stopped answering")
            if job.arm.model not in self.dead:
                self.dead.append(job.arm.model)
            self._stop("endpoint")
        else:
            raised = isinstance(outcome, tuple) and outcome[0] == "raised"
            how = f": {_hidden(str(outcome[1]), self.secrets)}" if raised else f" (exit code {exitcode})"
            self.died.append(f"the worker running {job.name()} stopped without finishing{how}")
            self._stop("worker")

    def finish(self) -> None:
        """Exit non-zero when an endpoint stopped answering or a worker died, after the runs going on have finished."""
        self.save()
        if self.dead:
            where = "; ".join(f"model {model!r}: {self.endpoints[model]}" for model in self.dead)
            raise SystemExit(f"An endpoint stopped answering ({where}). The arms that were running have finished, and "
                             "no other arm was started. Run again once it answers: the sweep resumes where it stopped.")
        if self.died:
            died = "; ".join(self.died)
            raise SystemExit(f"{died[:1].upper()}{died[1:]}. The rows written stand, and the next run resumes where it "
                             "stopped.")

    # --- the rounds ---------------------------------------------------------------------------------------------

    def first_round(self) -> None:
        """The screen, the check round, or a grid with no screen before it.

        Every start's first run comes first, so each room is known early. Under `auto` the start's second run is queued
        among the other arms, and once every start still searching has run twice the decision is recorded and, when it
        is 2, every other arm queued again. A known count queues each run of every arm in turn.
        """
        keys = [key for key in self.arms if self._active(key)]
        auto = self.cfg.sweep.repeats == "auto"
        jobs = [self.job(self.arms[key][0], 1) for key in keys]
        for key in keys:
            start, *others = self.arms[key]
            firsts = [self.job(arm, 1) for arm in others]
            if auto:
                firsts.insert(len(firsts) // 2, self.job(start, 2))
            jobs += firsts
        for repeat in range(2, self.state.get("repeats", 1) + 1):
            for key in keys:
                jobs += [self.job(arm, repeat) for arm in self.arms[key][1 if auto and repeat == 2 else 0:]]
        self.drain(deque(jobs), self._advance)

    def _advance(self, pending: deque[_Job]) -> None:
        """What the first round's finished runs decide: each start's room, once its first run is done, and the repeats,
        once every start still searching has run twice."""
        for key in self.arms:
            model, mode = key
            if (self._active(key) and self.compared[key] and mode not in self.state.get("room", {}).get(model, {})
                    and self.job(self.arms[key][0], 1).finished()):
                self._check_room(key)
        if self.cfg.sweep.repeats != "auto" or "repeats" in self.state:
            return
        keys = [key for key in self.arms if self._active(key)]
        if not keys or any(self.job(self.arms[key][0], repeat).left() for key in keys for repeat in (1, 2)):
            return
        decided = self.state["repeats"] = repeats_needed(self.cfg)
        self.save()
        _say("The start wrote the same SQL in both its runs, so every arm runs once." if decided == 1 else
             "The start's two runs did not show the same SQL for every case, so every arm runs twice.")
        if decided == 2:
            pending.extend(self.job(arm, 2) for key in keys for arm in self.arms[key][1:])

    def _check_room(self, key: tuple[str, str]) -> None:
        """Stop the search for `key`'s model and mode when its start left less room than an arm must gain to win: the
        smallest gain that passes the strictest Holm bar with the round's arm count, `detectable_gain`."""
        model, mode = key
        answerable = sum(case.answerable for case in self.parts["tuning"])
        misses = room(self.cfg, model, mode)
        needed = detectable_gain(answerable, self.compared[key], self.cfg.propose.alpha)
        record: dict = {"misses": misses, "needed": needed, "compared": self.compared[key]}
        if needed is None or misses < needed:
            record["stopped"] = "no room"
            gain = (f"{needed} more right" if needed is not None
                    else f"more than the {answerable} answerable questions there are")
            _say(f"{model}, {mode}: no room to search. The start missed {misses} of the {answerable} answerable "
                 f"questions to tune on, and with {self.compared[key]} arms compared with it a win takes {gain}, with "
                 "none lost, so no arm can win. No other arm of this model and mode starts. Check mode ([search] "
                 'strategy = "check") compares the start with the engine\'s defaults on every question, for harm.')
        self.state.setdefault("room", {}).setdefault(model, {})[mode] = record
        self._settle()
        self.save()

    def grid(self) -> None:
        """Every combination of the values the screen kept, for each model and mode still searching, each run as often
        as the first round decided, which is also the count the screen's report reads the kept values at. No control
        decides a kept value, so none is graded for it."""
        from mnemiq.sweep.report import kept_values, report  # the report reads this module's files

        keys = [key for key in self.arms if self._active(key) and self.compared[key]]
        if not keys:
            return
        repeats = self.state["repeats"]
        kept = kept_values(self.cfg, report(self.cfg, "screen", repeats=repeats, controls=False)[1])
        arms: list[Arm] = []
        refused: list[tuple[Arm, str]] = []
        for key in keys:
            for arm in grid_arms(self.cfg, {key: kept.get(key, {})}):
                if arm.round != "grid":
                    continue  # one change: the screen's own arm
                why = refusal_under(self.cfg, arm, self.deployments)
                if why is None:
                    arms.append(arm)
                else:
                    refused.append((arm, why))
        if refused:
            _say("Skipped, since the engine refuses them:", *_refusals(refused))
        self.drain(deque(self.job(arm, repeat) for repeat in range(1, repeats + 1) for arm in arms))


def _smoke(cfg: SweepConfig, count: int) -> None:
    """Each arm of the first round once, on the first `count` cases to tune on, in the out folder's smoke folder, which
    nothing reads: whether every arm runs, with no budget, room or repeats.

    Each smoke replaces the folder whole, but only one a smoke made, which holds its mark: anything else at that path,
    a link included, is refused and left as it is.
    """
    folder = Path(cfg.sweep.out) / SMOKE
    move = "Move it, or give [sweep] out another folder."
    if folder.is_symlink():
        raise SweepConfigError(f"{folder} is a symbolic link, which a smoke never makes, so it is left as it is. "
                               f"{move}")
    if folder.exists() and not folder.is_dir():
        raise SweepConfigError(f"{folder} is not a folder, so a smoke did not make it, and it is left as it is. {move}")
    if folder.exists() and not (folder / SMOKE_MARK).is_file():
        raise SweepConfigError(f"{folder} holds no mark of a smoke ({SMOKE_MARK}), so a smoke did not make it, and it "
                               f"is left as it is. {move}")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / SMOKE_MARK).write_text("A smoke made this folder, and the next smoke replaces it whole.\n",
                                     encoding="utf-8")
    smoke = cfg.model_copy(update={"sweep": cfg.sweep.model_copy(update={"out": str(folder)})})
    sweep = _Rounds(smoke, None, {"tuning": _parts(cfg)["tuning"][:count], "held_out": []})
    sweep.drain(deque(sweep.job(arm, 1) for arms in sweep.arms.values() for arm in arms))
    sweep.finish()


def run(cfg: SweepConfig, until: str | None = None, smoke: int | None = None) -> None:
    """Run the sweep's rounds in order: through `until` when it names one, or, with `smoke`, each arm of the first round
    on that many cases.

    A run holds the out folder's lock throughout, so a second run on the same folder is refused at once. Before anything
    runs, an identity the sweep cannot grade is refused (`identity_refusal`). Then the first round runs, and, after a
    screen, the grid of the values it kept. A search ends with the timing round and the confirm round
    (`mnemiq.sweep.propose.after_search`).

    A sweep resumes where it stopped, run after run:
    - `sweep.json` holds the fingerprint and the engine build the sweep started under. A run under another config (the
      config, a deployment's settings, or a pinned file, such as a store, an access policy, an env file's setting or a
      start profile) or another engine build is refused before any arm. A resumed config may change only what the
      fingerprint leaves out: the budgets, `seconds_per_question` and the controls.
    - So no arm's own record of its pinned files (`pins.json`) ever meets a resume under another set of files, which it
      would read as a move and mark the arm invalid for good.
    - The split is read back from `split.json`. Rows with no `sweep.json` or no `split.json` are refused.
    - A run answers only what each arm has not: an arm that stopped resumes from its rows. An arm `run_arm` found
      invalid is reported and never run again, and its cases in `retries.json` are no work left.
    - What the first round decided, the repeats and each model and mode's room, is read back, not measured again.

    `sweep.json` holds `fingerprint`, `engine_build` and `split_seed`; `repeats`, the runs each arm owes, once the first
    round has decided it or from the start when the config fixes it; `room`, by model and mode, the start's `misses`,
    the gain an arm `needed` (None when none can win), the arms `compared`, and `stopped: "no room"` when the search
    stopped there; `starts`, by model and mode, None where the engine runs the start, else why it refuses it, as the
    first round's plan found it (a mode once run stays so); `seconds`, the time the sweep's runs have taken; `stopped`,
    the sweep's own stop: `"budget"`, `"endpoint"` or `"worker"` (a worker that died), each cleared by the next run, or
    `"no room"` once every model and mode that searches has stopped for it; and `finished`, true once the strategy's
    last round has run, which a proposal waits for.

    A stop for the budget comes before an arm, never during one. When an endpoint stops answering, or a worker dies,
    the runs going on finish, none starts, and `run` exits non-zero naming the endpoint, or the run and what its worker
    raised.
    """
    strategy = cfg.search.strategy
    rounds = (*_ROUNDS_OF[strategy], *(_AFTER_SEARCH if strategy != "check" else ()))
    if until is not None and until not in rounds:
        raise SweepConfigError(f"{until!r} is not a round this sweep runs: with strategy {strategy!r} it runs "
                               f"{', '.join(rounds)}")
    if smoke is not None and (isinstance(smoke, bool) or not isinstance(smoke, int) or smoke < 1):
        raise SweepConfigError("a smoke asks each arm at least one case")
    out = Path(cfg.sweep.out)
    out.mkdir(parents=True, exist_ok=True)
    with _sole(out / OUT_LOCK, SweepConfigError(f"{out} is in use by another run of a sweep; run again once it has "
                                                "ended")):
        if (why := identity_refusal(cfg)) is not None:
            raise SweepConfigError(f"the sweep's identity cannot be graded: {why}")
        if smoke is not None:
            _smoke(cfg, smoke)
            return
        state = _begin(cfg)
        sweep = _Rounds(cfg, state, _parts(cfg))
        sweep.first_round()
        if not sweep.halted and until != sweep.first and strategy == "screen-then-grid":
            sweep.grid()
        if not sweep.halted and until in (None, "check", *_AFTER_SEARCH):
            from mnemiq.sweep.propose import after_search  # the proposal reads this module's files

            after_search(sweep, until)
        sweep.finish()
