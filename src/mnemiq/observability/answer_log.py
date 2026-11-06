"""A local record of what produced each answer, for a deployment with no sink.

With no `control_dsn` the metrics sink is a no-op and with no `verity_traces_url` there is no trace, so without this nothing
says which profile answered. Holds no question text, SQL or rows unless `MNEMIQ_ANSWER_LOG_TEXT` asks for the first two: rows
are the customer's data, and this file sits on a disk nobody decided to put them on. Never rotated yet.

Fail-soft like every sink: a log that cannot be written never stops an answer, and says so once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _git(package: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(package), *args], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def engine_commit(declared: str | None = None, *, package: Path | None = None) -> str | None:
    """The commit this engine runs, `-dirty` when its source was edited, None when that cannot be said.

    Only from the engine's own checkout. A wheel in a venv under another project sits inside that project's repo, and
    stamping its HEAD would be a wrong answer that looks like a right one. A build with no checkout declares it instead.
    """
    if declared:
        return declared
    if package is None:
        import mnemiq

        package = Path(mnemiq.__file__).resolve().parent
    top = _git(package, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() / "src" / "mnemiq" != package.resolve():
        return None
    head = _git(package, "rev-parse", "HEAD")
    if head is None:
        return None
    # The source only: an untracked note or a test beside it does not change what ran.
    dirty = _git(package, "status", "--porcelain", "--", str(package))
    return f"{head}-dirty" if dirty else head


class AnswerLog:
    """One JSON line per answer, and one per profile at boot, so a version can always be read back."""

    def __init__(self, path: str, *, include_text: bool = False) -> None:
        self.path = path
        self._include_text = include_text
        self._lock = threading.Lock()
        self._warned = False

    def _write(self, record: dict) -> None:
        line = json.dumps({"at": datetime.now(UTC).isoformat(), **record}, default=str, sort_keys=True) + "\n"
        try:
            with self._lock:
                # Private to its owner: it says who may see what, through the grant fingerprint.
                descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                with os.fdopen(descriptor, "a") as fh:
                    fh.write(line)
        except OSError as exc:
            if not self._warned:
                self._warned = True
                logger.warning("answer log %s cannot be written (%s): answers continue unrecorded", self.path, exc.strerror)

    def record_profile(self, name: str, version: str, view: dict, engine: str | None) -> None:
        self._write({"kind": "profile", "name": name, "version": version, "view": view, "engine_commit": engine})

    def record_answer(self, answer, question: str, total_ms: float) -> None:
        trace = answer.trace
        record = {
            "kind": "answer",
            # A correlation key, as in the trace: questions are enumerable, so this confirms a guess and hides little.
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "mode": answer.mode,
            "profile": answer.profile,
            "profile_version": answer.profile_version,
            "model": answer.model,
            "engine_commit": answer.engine_commit,
            "enrichment_version": getattr(trace, "enrichment_version", None),
            "grant_fingerprint": answer.grant_fingerprint,
            "deferred": answer.deferred,
            "failed": answer.failed,
            # The rows stand and their sentence failed: findable here, since the answer itself reads as a success.
            "prose_failed": getattr(answer, "prose_failed", False),
            "reason_code": str(answer.reason_code) if answer.reason_code else None,
            "cached": answer.cached,
            "total_ms": round(total_ms, 1),
        }
        if self._include_text:
            record["question"] = question
            # As asked: the executed statement carries the row policy's own predicate.
            record["sql"] = getattr(trace, "asked_sql", None) or getattr(trace, "plan_sql", None)
        self._write(record)


def build_answer_log(settings) -> AnswerLog | None:
    path = settings.answer_log_path
    if path == "off":
        return None
    if path is None:
        if settings.store_path == ":memory:":
            logger.info("no answer log: this store is in memory, so there is no path to keep one beside; set "
                        "MNEMIQ_ANSWER_LOG_PATH")
            return None
        path = f"{settings.store_path}.answers.jsonl"
    return AnswerLog(path, include_text=settings.answer_log_text)
