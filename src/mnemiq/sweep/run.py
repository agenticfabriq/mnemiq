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
final.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from openai import APIError

from mnemiq.agent.loop import AgentAnswer
from mnemiq.config import Settings
from mnemiq.contract import DeferralReason, EvaluationCase, IdentityContext
from mnemiq.eval.ablation import names_table
from mnemiq.eval.harness import CaseResult, Outcome, run_case
from mnemiq.llm.client import ModelUnavailable, model_error_kind
from mnemiq.observability.answer_log import engine_commit
from mnemiq.runtime import Runtime, _authz, build_runtime, load_current_snapshot
from mnemiq.store.bootstrap import init_store
from mnemiq.sweep.arms import Arm, baseline, profile_text
from mnemiq.sweep.config import (
    SweepConfig, SweepConfigError, _deployment_ids, deployment_settings, load_questions, pinned_files,
)

ROWS = "rows.jsonl"
PROFILE = "profile.toml"
INVALID = "invalid.json"
PINS = "pins.json"
RETRIES = "retries.json"
# Endpoint errors in a row that stop an arm.
_STREAK = 3
# Statuses that fail every question alike: a wrong or revoked key, a model name the server does not serve, a rate
# limit. Every 5xx joins them. Any other status is the one question's.
_ENDPOINT_STATUSES = frozenset({401, 403, 404, 429})
_STATUS = re.compile(r"status_(\d{3})")
# The one reason a failed answer gives when the source failed it: the source did not answer. Every other failed answer
# is the arm's own, `EXECUTION_FAILED` among them: the source rejected or timed out on every repair of the arm's SQL. A
# statement someone else cancelled reads the same way, which a store copy never sees.
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
                runtimes: dict[str | None, Runtime], build: dict) -> bool:
    """Ask every case and mode not yet in `rows.jsonl`; True when the endpoint stopped answering."""
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


def run_arm(cfg: SweepConfig, arm: Arm, cases: Sequence[EvaluationCase], folder: Path) -> str:
    """Run `arm` on `cases` into `folder`, resuming from the rows already there.

    An arm with an `invalid.json` is not run again. Its first run records the pinned files in `pins.json`; every run
    checks the files against that record, copies each store the cases read and checks each copy against it, writes the
    arm's `profile.toml`, and builds one runtime per deployment. Each case not yet answered is asked as the config's
    identity in each of the arm's modes, graded by `run_case`, and appended to `rows.jsonl` as one flushed row per case
    and mode. When the run ends, the pinned files are read again, and `invalid.json` records any that moved from the
    record.

    Returns `"invalid"` (a pinned file moved, in this run or before it), else `"endpoint"` (the endpoint stopped
    answering; the rows stand, and the next run resumes), else `"complete"`.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
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
    (folder / PROFILE).write_text(profile_text(cfg, arm), encoding="utf-8")
    runtimes: dict[str | None, Runtime] = {}
    try:
        for db_id in db_ids:
            runtimes[db_id] = build_runtime(arm_settings(cfg, arm, folder, db_id))
        stopped = _answer_all(cfg, arm, cases, folder, runtimes, build)
    finally:
        for runtime in runtimes.values():
            runtime.con.close()
    if moved := _changed_since(cfg, pinned):
        _write_json(folder / INVALID, moved)
        return "invalid"
    return "endpoint" if stopped else "complete"
