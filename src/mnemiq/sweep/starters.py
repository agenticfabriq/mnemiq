"""Starters, as a sweep reads them: the evidence a starter ships with, what the model server reports of its own build,
and where the build a sweep runs on differs from the one a starter was swept on.

A starter is a profiles file the engine ships (`mnemiq.profiles.starter_path`), and beside it its evidence,
`<name>.evidence.json`. The evidence holds builds and aggregate counts only, never a question, a statement, a case id or
a path. Its schema has no free-form table: every table is keyed by a mode, a part of the split or a database name of a
bounded form, and every value is a record or a scalar, so a key the schema does not name is refused at any depth.

What a server reports of its build is read from it, never assumed. A request that fails, a status other than 200, or a
field the server does not report leaves that field None, "not read", which is never a value: it is not compared, and a
later reading fills it in. The model itself is never read, only stated (`[models.<name>.build]`), since a server's
launch argument may be a repository id, an absolute path, a relative path or a cache snapshot, and no form tells them
apart. The last part of what the server was launched from is kept only to say when it is not the stated model's name.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mnemiq import profiles
from mnemiq.sweep.config import SweepConfig, SweepConfigError, start_starter
from mnemiq.sweep.run import engine_build

EVIDENCE_SUFFIX = ".evidence.json"
# What `server_readings` reads of a model server's build, each None when it was not read.
READINGS = ("server", "server_version", "context_window", "root_name")
# Each build request is read against a deadline and a size cap, so a proxy dripping a catch-all page cannot hold a run.
_READ_TIMEOUT_S = 5.0
_READ_BYTES = 1024 * 1024

Mode = Literal["instant", "thinking", "deep"]
Part = Literal["tuning", "held_out"]
Day = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
# A deployment's `db_id`, or "" for a sweep of one deployment: never a path.
DbKey = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{0,64}$")]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelBuildRecord(_Record):
    """The build a starter was swept on: the model, its quantization and the GPU as the config stated them, and the
    server, its version and its window as the server reported them."""

    model: str
    quantization: str
    server: str
    server_version: str
    gpu: str
    context_window: int


class EngineBuildRecord(_Record):
    """The engine the sweeps ran on: its package version, its commit when one could be read, and its source hash."""

    version: str
    commit: str | None
    source_sha: str


class DatasetRecord(_Record):
    name: str
    questions: int
    answerable: int


class RuleRecord(_Record):
    """The rule the sweeps ran under. Both bounds are 0, so the schema itself refuses a starter with looser ones."""

    strategy: Literal["check", "screen-then-grid", "grid"]
    alpha: float
    max_more_wrong: Literal[0]
    max_more_made_up: Literal[0]
    max_errors: int
    repeats: int


class Counts(_Record):
    """One run of an arm in one mode: the report's `Tally`, less the versions its rows were stamped with."""

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
    prompt_tokens: float | None


class CheckRun(_Record):
    run: int
    start: Counts
    defaults: Counts


class PartCounts(_Record):
    cases: int
    answerable: int


class HeldOut(_Record):
    start: list[Counts]
    candidate: list[Counts]


class SearchRecord(_Record):
    split: dict[Part, PartCounts]
    compared: dict[Mode, int]
    picks: dict[Mode, str]
    confirmed: bool | None
    held_out: dict[Mode, HeldOut] | None


class StarterEvidence(_Record):
    """What a starter ships with: what it was swept on, by which rule, and what the sweeps measured, in aggregate."""

    starter: str
    published: Day  # the local date the starter was written
    dataset: DatasetRecord
    made_up_measured: bool  # whether the measured questions hold one with no answer, which a made-up answer needs
    model_build: ModelBuildRecord
    engine_build: EngineBuildRecord
    safe_by: Literal["check", "search_from_defaults"]
    rule: RuleRecord
    modes_measured: list[Mode]
    unmeasured_modes: dict[Mode, list[str]]
    held_unset: list[str]
    check: dict[Mode, list[CheckRun]] | None
    search: SearchRecord | None
    versions: dict[DbKey, str]
    measured_versions: dict[Mode, dict[DbKey, str]] | None  # when the profile was measured mode by mode


def build_name(repo_id: str) -> str:
    """The name a repository id gives a build: the part after `/`, lowercased."""
    return repo_id.split("/", 1)[-1].lower()


def evidence_path(name: str) -> Path:
    """Where the starter `name`'s evidence ships: beside its file, in the starters folder. The engine's `starter_path`
    refuses a name the folder does not list."""
    return Path(profiles.starter_path(name)).with_name(f"{name}{EVIDENCE_SUFFIX}")


def _reasons(exc: ValidationError) -> str:
    """Where and why only: pydantic's own message quotes the input."""
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    return "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"]
                     for e in errors)


def read_evidence(name: str) -> StarterEvidence:
    """The evidence the starter `name` ships with, refused unless it is a regular file in the starters folder, as the
    starter's own file is, that the schema accepts and that names this starter."""
    try:
        path = evidence_path(name)
    except profiles.ProfileFileError as exc:
        raise SweepConfigError(str(exc)) from None
    if not path.exists() and not path.is_symlink():
        reason = f"{path.name} is missing"
    elif path.is_symlink() or not path.is_file() or path.resolve().parent != profiles.starters_folder().resolve():
        reason = f"{path.name} is not a regular file in the starters folder"
    else:
        try:
            evidence = StarterEvidence.model_validate_json(path.read_bytes())
        except OSError as exc:
            reason = f"{path.name} cannot be read: {exc.strerror}"
        except ValidationError as exc:
            reason = _reasons(exc)
        else:
            if evidence.starter == name:
                return evidence
            reason = f"it names starter {evidence.starter!r}"
    raise SweepConfigError(f"starter {name!r} ships without valid evidence ({reason})")


def _json(url: str, key: str | None) -> object | None:
    """What `url` answers with, read as JSON, or None for a failed request, a status other than 200, a reply past the
    deadline or the size cap, or a body that is not JSON. The key goes as a bearer token and is never logged."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    started = time.monotonic()
    try:
        with httpx.stream("GET", url, headers=headers, timeout=_READ_TIMEOUT_S, follow_redirects=False) as response:
            if response.status_code != 200:
                return None
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > _READ_BYTES or time.monotonic() - started > _READ_TIMEOUT_S:
                    return None
                chunks.append(chunk)
    except (httpx.HTTPError, httpx.InvalidURL):
        return None
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        return None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def server_readings(base_url: str, served: str, key: str | None) -> dict:
    """What the model server at `base_url` reports of its build, for the model it serves as `served` (the model's
    `llm_model`): `server` (the listing's `owned_by`), `server_version` (`/version`, at the URL's origin),
    `context_window` (`max_model_len`) and `root_name`, the last part of what the server was launched from.

    Only the one entry of `{base_url}/models` whose `id` is `served` is read, so another model the server lists is
    never taken for it. A field that could not be read is None, never a value, and the whole launch path is never
    returned: it names a folder on the server's machine."""
    readings: dict = dict.fromkeys(READINGS)
    listing = _json(f"{base_url.rstrip('/')}/models", key)
    data = listing.get("data") if isinstance(listing, dict) else None
    entries = [entry for entry in data if isinstance(entry, dict) and entry.get("id") == served] \
        if isinstance(data, list) else []
    if len(entries) == 1:
        (entry,) = entries
        window = entry.get("max_model_len")
        readings["server"] = _text(entry.get("owned_by"))
        readings["context_window"] = window if isinstance(window, int) and not isinstance(window, bool) else None
        if (root := _text(entry.get("root"))) is not None:
            readings["root_name"] = re.split(r"[\\/]+", root.strip("/\\"))[-1] or None
    parts = urlsplit(base_url)
    version = _json(f"{parts.scheme}://{parts.netloc}/version", key)
    readings["server_version"] = _text(version.get("version")) if isinstance(version, dict) else None
    return readings


def merged_readings(recorded: dict, now: dict, model: str | None = None) -> dict:
    """What a resumed sweep records of a model server's build: `recorded`, with each field it did not read and `now`
    read filled in. A field read both times that differs is refused, since the sweep's rounds would compare answers
    from two builds; a field either reading missed is never a difference. `model` names the server in the refusal."""
    merged = {field: recorded.get(field) for field in READINGS}
    for field in READINGS:
        before, value = recorded.get(field), now.get(field)
        if value is None:
            continue
        if before is None:
            merged[field] = value
        elif before != value:
            whose = f" for {model!r}" if model is not None else ""
            raise SweepConfigError(f"the model server{whose} changed since the sweep started: {field} was {before!r}, "
                                   f"now {value!r}. Serve the build the sweep started on, or start a new sweep in "
                                   "another out folder.")
    return merged


def _folded(value: object) -> str:
    """A value as builds are compared: casefolded, its spacing collapsed."""
    return " ".join(str(value).split()).casefold()


def build_mismatches(cfg: SweepConfig, model: str, readings: dict | None) -> list[str]:
    """Where the build `model` runs on differs from the one its start's starter was swept on, one line each, and one
    line naming what could not be compared: empty when the start reaches no starter, or when everything matches.

    `readings` is what `server_readings` read, or None before a run. The model, its quantization and the GPU are
    compared as `[models.<name>.build]` states them; the server, its version and its window as the server reports
    them; the config's `llm_context_window` against the window the starter was swept with; and the running engine
    against the one the starter was swept on, by source hash."""
    name = start_starter(cfg, model)
    if name is None:
        return []
    shipped = read_evidence(name)
    swept = shipped.model_build.model_dump()
    entry = cfg.models[model]
    stated = entry.build.model_dump() if entry.build is not None else {}
    read = readings or {}
    now = {"model": stated.get("model"), "quantization": stated.get("quantization"), "server": read.get("server"),
           "server_version": read.get("server_version"), "gpu": stated.get("gpu"),
           "context_window": read.get("context_window")}
    compared = [(field, swept[field], now[field]) for field in swept]
    compared.append(("llm_context_window", swept["context_window"], entry.llm_context_window))
    lines, unchecked = [], []
    for field, before, value in compared:
        if value is None:
            unchecked.append(field)
        elif _folded(before) != _folded(value):
            lines.append(f"starter {name!r}, model {model!r}: {field} differs: swept on {before!r}, this sweep has "
                         f"{value!r}")
    if unchecked:
        lines.append(f"starter {name!r}, model {model!r}: not checked, since the config does not state them or the "
                     f"server does not report them: {', '.join(unchecked)}")
    ran, running = shipped.engine_build, engine_build()
    if running["source_sha"] != ran.source_sha:
        lines.append(f"starter {name!r}: the engine differs: swept on {ran.source_sha[:12]} "
                     f"({ran.commit or 'no commit'}), running {running['source_sha'][:12]} "
                     f"({running['commit'] or 'no commit'})")
    return lines


def weights_folder_note(cfg: SweepConfig, model: str, readings: dict | None) -> str | None:
    """A line when the server reports weights launched from a folder whose name is not the stated model's (casefolded),
    else None. Said, never refused: a cache snapshot's folder is a hash."""
    entry = cfg.models[model]
    stated = entry.build.model if entry.build is not None else None
    root_name = (readings or {}).get("root_name")
    if stated is None or root_name is None or root_name.casefold() == build_name(stated).casefold():
        return None
    return (f"the server's weights folder for {model!r} is {root_name!r}, not {stated}'s name: check "
            f"[models.{model}.build] model")
