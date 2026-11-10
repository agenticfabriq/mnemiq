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

A starter is published (`publish_starter`, `mnemiq sweep starter`) from finished sweeps whose rule found it safe over
the engine's defaults: check mode's verdict, under either reading of its harm rule (`[propose] harm`, which the
evidence records), or a search from those defaults whose pick was confirmed on the held-out part. Every refusal comes
before anything is written. The evidence records the sha256 of the starter's file (`starter_sha256`), and the engine
loads a starter only beside evidence that does. The pair is staged outside the starters folder and checked there; only
then is each moved in, the evidence first, under a hidden name until its move and under the folder's lock, so the
folder never lists a draft.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import time
import shutil
import tomllib
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, ValidationError, model_serializer,
    model_validator,
)

from mnemiq import profiles
from mnemiq.agent.modes import MODES
from mnemiq.config import Settings
from mnemiq.observability.answer_log import engine_commit
from mnemiq.profiles import _NAME, ENDPOINT_SETTINGS, EVIDENCE_SUFFIX, PROFILE_SETTINGS
from mnemiq.sweep.arms import _endpoint, _toml, arm_of
from mnemiq.sweep.config import (
    PROPOSAL, Harm, SweepConfig, SweepConfigError, _RepoId, load_questions, start_profile, start_starter,
)
from mnemiq.sweep.run import OUT_LOCK, PINS, _sole, arm_folder, engine_build

# What `server_readings` reads of a model server's build, each None when it was not read.
READINGS = ("server", "server_version", "context_window", "root_name")
# What a resumed sweep refuses when the server now reports it otherwise. The weights folder's name is kept only to say,
# never to refuse: a cache snapshot's folder is a hash, and can move with the same weights in it.
_REFUSED_WHEN_CHANGED = ("server", "server_version", "context_window")
# Each build request is read against a deadline and a size cap, so a proxy dripping a catch-all page cannot hold a run.
_READ_TIMEOUT_S = 5.0
_READ_BYTES = 1024 * 1024

Mode = Literal["instant", "thinking", "deep"]
Part = Literal["tuning", "held_out"]


def _calendar_day(value: str) -> str:
    date.fromisoformat(value)  # refuses a month or a day no calendar has, such as 2026-99-99
    return value


# A day as `YYYY-MM-DD`, in ASCII digits (`\d` would take any script's), and one a calendar has.
Day = Annotated[str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"), AfterValidator(_calendar_day)]
# A deployment as published evidence holds it: numbered `deployment-<n>`, never named, since a deployment's own name
# (its `db_id`) can name a customer; or "" for a sweep of one deployment.
DeploymentKey = Annotated[str, Field(pattern=r"^(?:deployment-[1-9][0-9]*)?$")]
# Text a person states that a starter publishes: printable ASCII with no slash of either kind, so it is never a path,
# and at least one character that is not a space. The label of the questions (`--dataset`) takes at most
# `DATASET_LABEL_MAX` characters, and the GPU and the quantization `[models.<name>.build]` states at most `STATED_MAX`.
DATASET_LABEL_PATTERN = r"^[\x20-\x2e\x30-\x5b\x5d-\x7e]*[\x21-\x2e\x30-\x5b\x5d-\x7e][\x20-\x2e\x30-\x5b\x5d-\x7e]*$"
DATASET_LABEL_MAX = 80
DatasetLabel = Annotated[str, Field(pattern=DATASET_LABEL_PATTERN, max_length=DATASET_LABEL_MAX)]
STATED_MAX = 64
StatedText = Annotated[str, Field(pattern=DATASET_LABEL_PATTERN, max_length=STATED_MAX)]
# The model servers a starter may be published from, by the name each reports for itself (`owned_by`), which the
# evidence publishes. A closed list: a server joins it, one line here, once a starter is to be swept on it. A sweep on
# any other server still runs, and says where its build differs; only what may be published is held to this.
PUBLISHABLE_SERVERS: tuple[str, ...] = ("vllm",)
PublishableServer = Literal[PUBLISHABLE_SERVERS]
# A server's version as the evidence publishes it: a release as PEP 440 names its parts, digits and dots, then
# optionally a pre-release (`a`, `b` or `rc` and a number), a `.post` and a `.dev` number, and a local part only of the
# accelerator tags a build is for (`cu124`, `rocm6.2`, `cpu`, `tpu`, `xpu`, in any case, dot-separated). A word the
# server adds, such as an organisation's name in a local part, is never published, and neither is a slash, a backslash,
# a colon or a space.
_ACCELERATOR = r"(?:cu[0-9]+|rocm[0-9]+(?:\.[0-9]+)*|cpu|tpu|xpu)"
SERVER_VERSION_PATTERN = (r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
                          rf"(?:\+(?i:{_ACCELERATOR}(?:\.{_ACCELERATOR})*))?$")
SERVER_VERSION_MAX = 40
ServerVersion = Annotated[str, Field(pattern=SERVER_VERSION_PATTERN, max_length=SERVER_VERSION_MAX)]
# The sha256 of a starter's file, as its evidence records it.
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ModelBuildRecord(_Record):
    """The build a starter was swept on: the model, its quantization and the GPU as the config stated them, and the
    server, its version and its window as the server reported them."""

    model: _RepoId
    quantization: StatedText
    server: PublishableServer
    server_version: ServerVersion
    gpu: StatedText
    context_window: int


class EngineBuildRecord(_Record):
    """The engine the sweeps ran on: its package version, its commit when one could be read, and its source hash."""

    version: str
    commit: str | None
    source_sha: str


class DatasetRecord(_Record):
    name: DatasetLabel
    questions: int
    answerable: int


class RuleRecord(_Record):
    """The rule the sweeps ran under. Both bounds are 0, so the schema itself refuses a starter with looser ones.

    `harm` is how check mode read the bounds (`[propose] harm`), and "worst" is refused with any other strategy, as
    the sweep's config refuses it. It is written only when it is "worst", and read as "per_run" when absent, so
    evidence published before the reading existed reads, and is written, as it was."""

    strategy: Literal["check", "screen-then-grid", "grid"]
    alpha: float
    max_more_wrong: Literal[0]
    max_more_made_up: Literal[0]
    harm: Harm = "per_run"
    max_errors: int
    repeats: int

    @model_validator(mode="after")
    def _worst_in_check_mode_alone(self) -> RuleRecord:
        if self.harm == "worst" and self.strategy != "check":
            raise ValueError(f'harm "worst" is check mode\'s reading alone, and the strategy is {self.strategy!r}')
        return self

    @model_serializer(mode="wrap")
    def _per_run_unwritten(self, handler: SerializerFunctionWrapHandler) -> dict:
        written = handler(self)
        if self.harm == "per_run":
            written.pop("harm", None)   # a dump told to leave harm out, or its default, has already left it out
        return written


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
    starter_sha256: Sha256  # the starter's file, which the engine loads only beside evidence that records it
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
    versions: dict[DeploymentKey, str]
    measured_versions: dict[Mode, dict[DeploymentKey, str]] | None  # when the profile was measured mode by mode


def build_name(repo_id: str) -> str:
    """The name a repository id gives a build: the part after `/`, lowercased."""
    return repo_id.split("/", 1)[-1].lower()


def evidence_path(name: str, folder: Path | None = None) -> Path:
    """Where the starter `name`'s evidence ships: beside its file, in the starters folder (or in `folder`). The engine's
    `starter_path` refuses a name the folder does not list."""
    return Path(profiles.starter_path(name, folder)).with_name(f"{name}{EVIDENCE_SUFFIX}")


def _reasons(exc: ValidationError) -> str:
    """Where and why only: pydantic's own message quotes the input."""
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    return "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"]
                     for e in errors)


def read_evidence(name: str, folder: Path | None = None) -> StarterEvidence:
    """The evidence the starter `name` ships with (or that `folder` holds beside it), refused unless it is a regular
    file in the starters folder, as the starter's own file is, that the schema accepts, that names this starter, and
    that records the sha256 of the starter's file."""
    try:
        path = evidence_path(name, folder)
        starter = Path(profiles.starter_path(name, folder))
    except profiles.ProfileFileError as exc:
        raise SweepConfigError(str(exc)) from None
    home = (profiles.starters_folder() if folder is None else Path(folder)).resolve()
    if not path.exists() and not path.is_symlink():
        reason = f"{path.name} is missing"
    elif path.is_symlink() or not path.is_file() or path.resolve().parent != home:
        reason = f"{path.name} is not a regular file in the starters folder"
    else:
        try:
            evidence = StarterEvidence.model_validate_json(path.read_bytes())
            held = hashlib.sha256(starter.read_bytes()).hexdigest()
        except OSError as exc:
            reason = f"{path.name} or {starter.name} cannot be read: {exc.strerror}"
        except ValidationError as exc:
            reason = _reasons(exc)
        else:
            if evidence.starter != name:
                reason = f"it names starter {evidence.starter!r}"
            elif evidence.starter_sha256 != held:
                reason = f"its starter_sha256 is not the sha256 of {starter.name}"
            else:
                return evidence
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
    read filled in. The server, its version or its window read both times and different is refused, since the sweep's
    rounds would compare answers from two builds; a field either reading missed is never a difference. The weights
    folder's name takes the latest reading and is never refused (`moved_weights_note` says it). `model` names the
    server in the refusal."""
    merged = {field: recorded.get(field) for field in READINGS}
    for field in READINGS:
        before, value = recorded.get(field), now.get(field)
        if value is None:
            continue
        if before is None or field not in _REFUSED_WHEN_CHANGED:
            merged[field] = value
        elif before != value:
            whose = f" for {model!r}" if model is not None else ""
            raise SweepConfigError(f"the model server{whose} changed since the sweep started: {field} was {before!r}, "
                                   f"now {value!r}. Serve the build the sweep started on, or start a new sweep in "
                                   "another out folder.")
    return merged


def moved_weights_note(recorded: dict, now: dict, model: str) -> str | None:
    """A line when the server's weights folder, read both times, is not the one the sweep recorded, else None."""
    before, value = recorded.get("root_name"), now.get("root_name")
    if before is None or value is None or before == value:
        return None
    return (f"the server's weights folder for {model!r} changed since the sweep started: was {before!r}, now "
            f"{value!r}. Said, never refused: check [models.{model}.build] model")


def _folded(value: object) -> str:
    """A value as builds are compared: casefolded, its spacing collapsed."""
    return " ".join(str(value).split()).casefold()


def build_mismatches(cfg: SweepConfig, model: str, readings: dict | None, engine: dict | None = None) -> list[str]:
    """Where the build `model` runs on differs from the one its start's starter was swept on, one line each, and one
    line naming what could not be compared: empty when the start reaches no starter, or when everything matches.

    `readings` is what `server_readings` read, or None before a run. The model, its quantization and the GPU are
    compared as `[models.<name>.build]` states them; the server, its version and its window as the server reports
    them; the config's `llm_context_window` against the window the starter was swept with; and `engine` (an
    `engine_build`: the one the sweep's arms ran on, for a proposal; by default the one running now) against the one the
    starter was swept on, by source hash."""
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
    ran, running = shipped.engine_build, engine if engine is not None else engine_build()
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


# --- publishing a starter ---------------------------------------------------------------------------------------------

# A starter's description, the same for every one, so no person types what its evidence says.
_DESCRIPTION = ("{name}: settings found safe over the engine's defaults on {dataset}; the evidence is in "
                "{name}" + EVIDENCE_SUFFIX)
# What a starter never takes from the proposal it is written from: where the model is served and judged, the key
# variables, who may pick the profile, and its description, which `_DESCRIPTION` replaces.
_NOT_CARRIED = frozenset({*ENDPOINT_SETTINGS, "api_key_env", "verify_api_key_env", "roles", "description"})
# The lock a publisher holds on the starters folder, so a second one is refused at once rather than moving its files in
# between the first one's.
PUBLISH_LOCK = ".publish.lock"
# The hidden copy `_move_in` makes of a starter's file or its evidence on its way into the starters folder, and nothing
# else: `.<name>.toml.<uuid>.tmp` or `.<name>.evidence.json.<uuid>.tmp`, the name as a profile name is written and the
# uuid as 32 lowercase hex digits. Only these are ever cleared, so another hidden file in the folder is left alone.
_HIDDEN_COPY = re.compile(rf"\.(?:{_NAME.pattern})(?:\.toml|{re.escape(EVIDENCE_SUFFIX)})\.[0-9a-f]{{32}}\.tmp")
_UNRESOLVED = "the starter does not resolve to the profile that was measured"
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _runs(count: int) -> str:
    return "1 run" if count == 1 else f"{count} runs"


def fewer_right(start: dict[str, list[int]], defaults: dict[str, list[int]]) -> list[str]:
    """One line for each mode, in the engine's order, where the starter's lowest right count over its runs (`start`, a
    mode to each run's count) is below the defaults' lowest over theirs (`defaults`): with one run of the defaults,
    any shortfall. Empty when no mode falls short. A shortfall in one run alone, whose lowest is not below the
    defaults' lowest, is not one."""
    return [f"in {mode} its lowest right count over {_runs(len(start[mode]))} is {min(start[mode])}, and the "
            f"defaults' lowest over {_runs(len(defaults[mode]))} is {min(defaults[mode])}"
            for mode in MODES if mode in start and min(start[mode]) < min(defaults[mode])]


def _key(name: str) -> str:
    """A setting's name as a TOML key: bare when TOML reads it bare, else quoted (`arms._toml`)."""
    return name if _BARE_KEY.fullmatch(name) else _toml(name)


def starter_text(proposal_text: str, name: str, dataset: str) -> str:
    """The starter `name`, from the text of a proposal's profiles file: its one profile, less what a starter never holds
    (`_NOT_CARRIED`), renamed `name`, its file's default, and described by `_DESCRIPTION`. Every name and string is
    written by `arms._toml`, so a dotted build name is a quoted key. Refused unless it reads back as that one profile
    and nothing else."""
    try:
        raw = tomllib.loads(proposal_text)
        held = {key: value for key, value in raw["profiles"][raw["default"]].items() if key not in _NOT_CARRIED}
    except (tomllib.TOMLDecodeError, KeyError, TypeError, AttributeError):
        raise SweepConfigError("the proposal's profiles file does not hold one profile that is its default") from None
    sections = {key: held.pop(key) for key in [key for key in held if key in MODES]}
    description = _DESCRIPTION.format(name=name, dataset=dataset)
    profile = _toml(name)
    try:
        lines = [f"default = {profile}", "", f"[profiles.{profile}]", f"description = {_toml(description)}"]
        lines += [f"{_key(key)} = {_toml(value)}" for key, value in held.items()]
        for mode, section in sections.items():
            lines += ["", f"[profiles.{profile}.{mode}]"]
            lines += [f"{_key(key)} = {_toml(value)}" for key, value in section.items()]
    except (TypeError, AttributeError):   # a value no profile setting takes, such as a table outside a mode's section
        raise SweepConfigError("the proposal's profile holds a value a starter cannot write") from None
    text = "\n".join(lines) + "\n"
    if tomllib.loads(text) != {"default": name, "profiles": {name: {"description": description, **held, **sections}}}:
        raise SweepConfigError(f"the proposal could not be written as starter {name!r} without changing what it holds")
    return text


def _proposal(cfg: SweepConfig, model: str) -> tuple[Path, Path]:
    """Where a sweep's proposal for `model` is: its profiles file and its evidence."""
    from mnemiq.sweep.propose import EVIDENCE, PROFILES  # the proposal reads this module

    folder = Path(cfg.sweep.out) / PROPOSAL / model
    return folder / PROFILES, folder / EVIDENCE


@dataclass(frozen=True)
class _Swept:
    """A finished sweep as publishing reads it, each file once, so every check and the evidence written read the same
    bytes: its config, its record (`sweep.json`), and its proposal for the model, the evidence parsed and the profiles
    file as it was read (None when nothing was proposed)."""

    cfg: SweepConfig
    state: dict
    evidence: dict
    profile: bytes | None


def _proposed(cfg: SweepConfig, model: str, state: dict) -> _Swept:
    """The sweep's proposal for `model`, read once, refused unless it is this sweep's and the rule chose it with both
    bounds at 0, under the reading of the harm rule (`[propose] harm`) the config fixed."""
    profiles_path, path = _proposal(cfg, model)
    try:
        evidence = json.loads(path.read_bytes())
    except (OSError, ValueError):
        raise SweepConfigError(f"{path} cannot be read: propose from the sweep before publishing a starter from "
                               "it") from None
    if not isinstance(evidence, dict) or (evidence.get("sweep") or {}).get("fingerprint") != state["fingerprint"]:
        raise SweepConfigError(f"{path} was not proposed from the sweep in {cfg.sweep.out}: propose from it again")
    if evidence.get("chosen_by") != "rule":
        raise SweepConfigError("the proposal was chosen by a person, and a starter is only what the rule proposes")
    for bound in ("max_more_wrong", "max_more_made_up"):
        if (value := evidence["rule"][bound]) != 0:
            raise SweepConfigError(f"[propose] {bound} is {value}: a starter needs both bounds at 0")
    # The starter records the reading the evidence holds, so it must be the one the config fixed: absent, "per_run".
    if (said := evidence["rule"].get("harm", "per_run")) != cfg.propose.harm:
        raise SweepConfigError(f"{path} records harm {said!r}, and the config the sweep ran under has [propose] harm "
                               f"{cfg.propose.harm!r}: propose from the sweep again")
    try:
        profile = profiles_path.read_bytes()
    except FileNotFoundError:
        profile = None
    except OSError as exc:
        raise SweepConfigError(f"{profiles_path} cannot be read: {exc.strerror}") from None
    return _Swept(cfg, state, evidence, profile)


def _paired(check: _Swept, search: _Swept | None) -> None:
    """Refuse a `--search` that is not a search, and a sweep published beside one that is not check mode: with
    `--search`, the sweep published is the check that started from the search's proposal."""
    if search is None:
        return
    if search.evidence["rule"]["strategy"] == "check":
        raise SweepConfigError(f"--search names a sweep in check mode ({search.cfg.path}), and it names the search "
                               "whose proposal the check started from")
    if (strategy := check.evidence["rule"]["strategy"]) != "check":
        raise SweepConfigError(f"--search is given, so the sweep published must be a check started from the search's "
                               f"proposal, and {check.cfg.path} ran strategy {strategy!r}: publish a search from the "
                               "engine's defaults without --search")


def _basis(cfg: SweepConfig, model: str, evidence: dict) -> str:
    """What the starter is safe by: check mode's verdict over the engine's defaults, or a search from those defaults
    whose pick was confirmed on the held-out part."""
    chosen = evidence["chosen"]
    if evidence["rule"]["strategy"] == "check":
        if chosen == "start":
            return "check"
        if chosen == "defaults":
            raise SweepConfigError("the rule proposed the engine's defaults: there is no starter to publish")
        raise SweepConfigError("the rule proposed no profile: there is no starter to publish")
    if start_profile(cfg, model) != (None, None):
        raise SweepConfigError("the search did not start from the engine's defaults: check its proposal against them "
                               "in check mode")
    if evidence["confirmed"] is not True:
        raise SweepConfigError("the search's pick was not confirmed on the held-out part")
    if (evidence.get("confirm") or {}).get("same_as_start"):
        # Every pick's change was lost to a conflict between the modes, so what was confirmed is the start, which is
        # the engine's defaults.
        raise SweepConfigError("the search's pick is the engine's defaults: there is no starter to publish")
    return "search_from_defaults"


def _right_counts(numbers: dict, arm: str) -> dict[str, list[int]]:
    """Each measured mode's right count in each run of `arm`, from check mode's evidence."""
    return {mode: [run["right"] for run in found[arm]] for mode, found in numbers.items()}


_STATED = ("model", "quantization", "gpu")
_READ_FROM_SERVER = ("server", "server_version", "context_window")


def _build_of(cfg: SweepConfig, model: str, state: dict) -> dict:
    """`model`'s build as the config states it and as the sweep's record read it from the server, each field None where
    it was not."""
    stated = cfg.models[model].build
    read = (state.get("servers") or {}).get(model) or {}
    return ({field: getattr(stated, field) if stated is not None else None for field in _STATED}
            | {field: read.get(field) for field in _READ_FROM_SERVER})


def _whole_build(model: str, sweeps: list[tuple[SweepConfig, dict]]) -> None:
    """Refuse, in this order: a build the check's config (the first of `sweeps`) does not state whole, or a server a
    sweep did not read whole; two sweeps whose servers read otherwise, or whose configs state the build otherwise where
    both state it; and a stated `llm_context_window` that is not the server's window. The weights folder's name is
    never part of it."""
    (cfg, state), *rest = sweeps
    build = _build_of(cfg, model, state)
    unread = {field for other, other_state in rest for field in _READ_FROM_SERVER
              if _build_of(other, model, other_state)[field] is None}
    if missing := [field for field in ModelBuildRecord.model_fields if build[field] is None or field in unread]:
        raise SweepConfigError(f"the model build is incomplete: {', '.join(missing)} not read from the server or not "
                               f"stated in [models.{model}.build]")
    for other, other_state in rest:
        found = _build_of(other, model, other_state)
        for field in _READ_FROM_SERVER:
            if found[field] != build[field]:
                raise SweepConfigError(f"the two sweeps read different servers: {field} {build[field]!r} and "
                                       f"{found[field]!r}")
        for field in _STATED:
            if found[field] is not None and found[field] != build[field]:
                raise SweepConfigError(f"the two sweeps state different builds in [models.{model}.build]: {field} "
                                       f"{build[field]!r} and {found[field]!r}")
    for each, _ in sweeps:
        window = each.models[model].llm_context_window
        if window is not None and window != build["context_window"]:
            raise SweepConfigError(f"[models.{model}] llm_context_window is {window}, and the server reports a "
                                   f"context_window of {build['context_window']}: state the server's window, or leave "
                                   "it out")


def _fits(pattern: str, longest: int, value: object) -> bool:
    return isinstance(value, str) and len(value) <= longest and re.fullmatch(pattern, value) is not None


def _publishable_build(model: str, build: dict) -> None:
    """Refuse a build whose published fields could carry anything but a build's facts: a server not on the closed list
    (`PUBLISHABLE_SERVERS`), a server version not shaped as a release (`SERVER_VERSION_PATTERN`), and a stated GPU or
    quantization that could carry a path (`StatedText`). The evidence publishes each as read or stated."""
    if build["server"] not in PUBLISHABLE_SERVERS:
        raise SweepConfigError(f"server {build['server']!r} is not a model server a starter may be published from "
                               f"({', '.join(PUBLISHABLE_SERVERS)}): what the server reports of itself is published")
    if not _fits(SERVER_VERSION_PATTERN, SERVER_VERSION_MAX, build["server_version"]):
        raise SweepConfigError(f"server_version {build['server_version']!r} is not a release version (digits and "
                               "dots, then optionally a pre-release such as rc2, .postN, .devN and an accelerator tag "
                               f"such as +cu124, in at most {SERVER_VERSION_MAX} characters): what the server reports "
                               "of itself is published")
    for field in ("gpu", "quantization"):
        if not _fits(DATASET_LABEL_PATTERN, STATED_MAX, build[field]):
            raise SweepConfigError(f"[models.{model}.build] {field} {build[field]!r} is not text a starter can "
                                   f"publish: 1 to {STATED_MAX} printable ASCII characters, not all spaces, with no / "
                                   "or \\")


def _started_from(cfg: SweepConfig, model: str, evidence: dict, search: _Swept) -> None:
    """Refuse a search whose proposal, as read, is not the file the check's start arm pinned as its start profile, by
    sha256: what the check's arms read, not the file as it is now."""
    proposal = _proposal(search.cfg, model)[0]
    start = start_profile(cfg, model)[0]
    first = next(iter(evidence.get("check") or {}), cfg.modes[0])
    pins = arm_folder(cfg, arm_of(model, first, {}, round="check"), 1) / PINS
    try:
        pinned = json.loads(pins.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pinned = {}
    found = hashlib.sha256(search.profile).hexdigest() if search.profile is not None else None
    if start is None or found is None or not isinstance(pinned, dict) or pinned.get(start) != found:
        raise SweepConfigError(f"the search's proposal {proposal} is not the file the check started from")


def _engine(build: dict) -> str:
    return f"{str(build.get('source_sha'))[:12]} ({build.get('commit') or 'no commit'})"


def _counts(run: dict) -> Counts:
    return Counts(**{field: run[field] for field in Counts.model_fields})


def _numbered(cfg: SweepConfig) -> dict[str, str]:
    """Each deployment the sweep read, by its `db_id`, to the key published evidence gives it: `deployment-<n>`, in the
    order of the sorted ids. One mapping serves every version the evidence holds. Empty for a sweep of one."""
    return {db_id: f"deployment-{number}"
            for number, db_id in enumerate(sorted(cfg.dataset.deployments or {}), start=1)}


def _by_deployment(version: str | dict, numbered: dict[str, str]) -> dict[str, str]:
    """A version as published evidence holds it, for each deployment: under "" for a sweep of one, else under each
    deployment's number (`numbered`), never its name. Refused when it names a deployment the sweep did not read, which
    no number could stand for."""
    if isinstance(version, str):
        return {"": version}
    if not set(version) <= set(numbered):
        raise SweepConfigError("the proposal's versions name a deployment the sweep did not read, so they cannot be "
                               "numbered: propose from the sweep again")
    return {numbered[db_id]: each for db_id, each in sorted(version.items())}


def _search_record(evidence: dict) -> SearchRecord:
    """What a search's proposal shows: its split, how many arms each mode compared with the start, each mode's pick,
    whether the pick was confirmed, and the held-out runs of the start and of the pick."""
    numbers = evidence.get("numbers")
    held_out = None if numbers is None else {
        mode: HeldOut(start=[_counts(run) for run in found["held_out"]["start"] or []],
                      candidate=[_counts(run) for run in found["held_out"]["arm"] or []])
        for mode, found in numbers.items()}
    return SearchRecord(split={part: PartCounts(cases=counts["cases"], answerable=counts["answerable"])
                               for part, counts in evidence["split"].items()},
                        compared=dict(evidence["holm"]), picks=dict(evidence["picks"]),
                        confirmed=evidence["confirmed"], held_out=held_out)


def _field_default(name: str) -> object:
    return Settings.model_fields[name].get_default(call_default_factory=True)


def starter_evidence(cfg: SweepConfig, model: str, name: str, dataset: str, *, state: dict, evidence: dict,
                     text: str, searched: dict | None) -> StarterEvidence:
    """The evidence the starter `name` ships with, field by field from what each source may give: builds, the rule,
    counts and versions, never a question, a statement, a case id or a path. Each source is passed as publishing read it
    once: the sweep `cfg`'s record (`state`) and its proposal's evidence for `model`, the starter's own `text`, and the
    evidence of the search behind it (`searched`: a check's `--search`, or the sweep itself when it is that search;
    None for check mode alone). Refused, never raised as the schema's own error, when it does not have the schema's
    form."""
    try:
        return _evidence(cfg, model, name, dataset, state, evidence, text, searched)
    except ValidationError as exc:
        raise SweepConfigError(f"the starter's evidence does not have the form the engine reads: "
                               f"{_reasons(exc)}") from None


def _evidence(cfg: SweepConfig, model: str, name: str, dataset: str, state: dict, evidence: dict, text: str,
              found: dict | None) -> StarterEvidence:
    """`starter_evidence` itself, which the schema may refuse. `text` is the starter's file as written, in UTF-8."""
    checked = evidence["rule"]["strategy"] == "check"
    held = tomllib.loads(text)["profiles"][name]
    wide = {key: value for key, value in held.items() if key in PROFILE_SETTINGS}
    measured = list(evidence["check"]) if checked else list(evidence["confirm"]["modes"])
    cases = load_questions(cfg)
    entry, read = cfg.models[model], state["servers"][model]
    stated = entry.build.model_dump()
    measured_versions = evidence.get("measured_versions")
    numbered = _numbered(cfg)
    return StarterEvidence(
        starter=name,
        starter_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        published=date.today().isoformat(),
        dataset=DatasetRecord(name=dataset, questions=len(cases), answerable=sum(case.answerable for case in cases)),
        made_up_measured=any(not case.answerable for case in cases),
        model_build=ModelBuildRecord(model=stated["model"], quantization=stated["quantization"], server=read["server"],
                                     server_version=read["server_version"], gpu=stated["gpu"],
                                     context_window=read["context_window"]),
        engine_build=EngineBuildRecord(version=importlib.metadata.version("mnemiq"),
                                       commit=state["engine_build"].get("commit"),
                                       source_sha=state["engine_build"]["source_sha"]),
        safe_by="check" if checked else "search_from_defaults",
        # The reading check mode passed under: a proposal's evidence records it only when it is "worst".
        rule=RuleRecord(**{field: evidence["rule"][field] for field in RuleRecord.model_fields if field != "harm"},
                        harm=evidence["rule"].get("harm", "per_run")),
        modes_measured=measured,
        unmeasured_modes={mode: sorted(key for key, value in wide.items() if value != _field_default(key))
                          for mode in MODES if mode not in measured},
        held_unset=sorted(PROFILE_SETTINGS - set(ENDPOINT_SETTINGS) - set(held)),
        check=None if not checked else {
            mode: [CheckRun(run=start["run"], start=_counts(start), defaults=_counts(defaults))
                   for start, defaults in zip(found_mode["start"], found_mode["defaults"], strict=True)]
            for mode, found_mode in evidence["check"].items()},
        search=_search_record(found) if found is not None else None,
        versions=_by_deployment(evidence["version"], numbered),
        measured_versions=None if measured_versions is None else {
            mode: _by_deployment(version, numbered) for mode, version in measured_versions.items()},
    )


def _deployment_text(cfg: SweepConfig, model: str, starter: str) -> str:
    """A deployment's profiles file whose default starts from the starter `starter`, with the model's endpoint and key
    variables from `[models]`, as the sweep's arms were served."""
    entry = cfg.models[model]
    judge_key = (cfg.judge if cfg.judge is not None else entry).verify_api_key_env
    lines = ['default = "publishing"', "", "[profiles.publishing]", f"starter = {_toml(starter)}",
             f"api_key_env = {_toml(entry.api_key_env)}"]
    lines += [f"verify_api_key_env = {_toml(judge_key)}"] if judge_key else []
    lines += [f"{field} = {_toml(value)}" for field, value in _endpoint(cfg, model).items() if value is not None]
    return "\n".join(lines) + "\n"


def _resolves_as_measured(cfg: SweepConfig, model: str, starter: str, version: str | dict, staged: Path) -> None:
    """Refuse unless a deployment whose profile starts from the starter `starter`, read from the folder it is staged in
    (`staged`), resolves, on every deployment the sweep read, to the version the proposal recorded: the engine's own
    loader and version function, as a run stamps its rows."""
    from mnemiq.sweep.propose import _agrees, _versions  # the proposal reads this module

    drafted = {None: version} if isinstance(version, str) else dict(version)
    with tempfile.TemporaryDirectory(prefix="mnemiq-starter-") as scratch:
        path = Path(scratch) / "profiles.toml"
        path.write_text(_deployment_text(cfg, model, starter), encoding="utf-8")
        try:
            resolved = _versions(cfg, path, {}, starters=staged)
        except SweepConfigError as exc:
            raise SweepConfigError(f"{_UNRESOLVED}: {exc}") from None
    if set(resolved) != set(drafted) or not _agrees(resolved, drafted):
        differs = "; ".join(f"{'the deployment' if db_id is None else f'the deployment for {db_id!r}'} gives it "
                            f"{resolved.get(db_id)}, and the proposal's version is {drafted.get(db_id)}"
                            for db_id in sorted(set(resolved) | set(drafted), key=str)
                            if resolved.get(db_id) != drafted.get(db_id))
        raise SweepConfigError(f"{_UNRESOLVED}: {differs}")


def publish_starter(cfg: SweepConfig, model: str, dataset: str, *, name: str | None = None,
                    search: SweepConfig | None = None) -> tuple[Path, Path]:
    """Write the starter for `model`'s build and its evidence into the starters folder (`profiles.starters_folder`),
    from the finished sweep `cfg`, whose rule found it safe over the engine's defaults: check mode's verdict, under
    either reading of its harm rule, which the evidence records, or a search from those defaults whose pick was
    confirmed. `search` is the search whose proposal the check started from, when it did; a check started from a
    profile of its own is published without one. `dataset` labels the questions, for the description and the
    evidence. Returns the two paths.

    Every refusal comes before anything is written, in this order: a dataset label that is not 1 to 80 printable ASCII
    characters, not all spaces, without a slash; an out folder a run or a proposal holds, whose lock publishing then
    holds to its end, as a proposal does; a key variable not set; a sweep that has not finished, or ran under another
    config; an engine with no checkout to write into; an engine other than the one the sweeps ran on, by source hash; a
    proposal a person chose, one whose bounds are not both 0, or one whose evidence records another reading of the harm
    rule than its config, a `search` in check mode, or a `search` beside a sweep that is not; a check that chose the
    defaults, or a search that did not start from them, whose pick was not confirmed, or whose pick is the defaults; a
    starter that gets fewer right answers than the defaults (`fewer_right`); a build not read and stated whole; a build
    whose server, server version, GPU or quantization is not what may be published (`_publishable_build`); a name that
    is not the build's; a search that is not the one the check started from; evidence the schema refuses; the staged
    pair's own checks, below; and the starters folder, while another publisher holds its lock. Each sweep's record and
    proposal are read once, so the checks and the evidence see the same bytes. Until every check has passed, nothing is
    written in the starters folder, not even the lock file.

    The evidence records the sha256 of the starter's file (`starter_sha256`), and the engine loads a starter only
    beside evidence that does. The two files are staged outside the starters folder and checked there: the evidence as
    the sweep reads it, and a deployment starting from the staged starter must resolve, on every deployment the sweep
    read, to the version the proposal recorded. Only then does publishing take the folder's lock, remove any hidden copy
    a killed publish left there, and move each file in, the evidence first, under a hidden name until its move. A
    failure before the moves leaves the starters folder as it was. One between them leaves a pair the engine refuses
    until both files are restored or published again."""
    from mnemiq.sweep.propose import _finished  # the proposal reads this module

    if len(dataset) > DATASET_LABEL_MAX or not re.fullmatch(DATASET_LABEL_PATTERN, dataset):
        raise SweepConfigError(f"--dataset {dataset!r} is not a label a starter can carry: it names the questions in "
                               f"the starter's description and evidence, so it is 1 to {DATASET_LABEL_MAX} printable "
                               "ASCII characters, not all spaces, with no / or \\")
    sweeps = [cfg] if search is None else [cfg, search]
    for each in sweeps:
        if model not in each.models:
            raise SweepConfigError(f"{each.path}: model {model!r} is not one this sweep ran; it ran "
                                   f"{', '.join(each.models)}")
    with ExitStack() as held:
        locked: list[Path] = []
        for each in sweeps:
            out = Path(each.sweep.out).resolve()
            if out in locked:
                continue
            if not out.is_dir():
                _finished(each)  # refuses: no sweep has run in that folder
            held.enter_context(_sole(out / OUT_LOCK, SweepConfigError(
                f"{out} is in use by a run or a proposal of this sweep; publish again once it has ended")))
            locked.append(out)
        return _publish(cfg, model, dataset, name, search)


def _clear_stale_copies(folder: Path) -> None:
    """Remove each hidden copy (`_HIDDEN_COPY`) that a publish killed between its copy and its move left in the starters
    folder, and no other file. Called with the folder's lock held, when no other publish can be moving one."""
    for path in folder.glob(".*.tmp"):
        if _HIDDEN_COPY.fullmatch(path.name) and (path.is_symlink() or path.is_file()):
            path.unlink(missing_ok=True)


def _move_in(source: Path, folder: Path) -> None:
    """Copy `source` into `folder` under a hidden name that ends in neither `.toml` nor the evidence's suffix, so no
    reader lists it, then move it into place under its own name: the folder never holds part of the file under that
    name. The hidden copy is removed on every path."""
    hidden = folder / f".{source.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(source, hidden)
        os.replace(hidden, folder / source.name)
    finally:
        hidden.unlink(missing_ok=True)


def _publish(cfg: SweepConfig, model: str, dataset: str, name: str | None,
             search: SweepConfig | None) -> tuple[Path, Path]:
    """`publish_starter` itself, with each sweep's out folder lock held. The starters folder's lock is taken once every
    check has passed, and held through staging, its checks and both moves."""
    from mnemiq.sweep.propose import _finished, _keys_set  # the proposal reads this module

    sweeps = [cfg] if search is None else [cfg, search]
    for each in sweeps:
        _keys_set(each, [model])
    states = [_finished(each) for each in sweeps]
    if engine_commit() is None:
        raise SweepConfigError("publishing writes into the engine's source: run it from a checkout")
    running = engine_build()
    for state in states:
        ran = state.get("engine_build") or {}
        if ran.get("source_sha") != running["source_sha"]:
            raise SweepConfigError(f"the sweep ran on engine {_engine(ran)}, and this is {_engine(running)}: publish "
                                   "from a checkout at the sweep's commit")
    swept = [_proposed(each, model, state) for each, state in zip(sweeps, states, strict=True)]
    checked, found = swept[0], swept[1] if search is not None else None
    _paired(checked, found)
    evidence = checked.evidence
    safe_by = _basis(cfg, model, evidence)
    if safe_by == "check" and (lines := fewer_right(_right_counts(evidence["check"], "start"),
                                                    _right_counts(evidence["check"], "defaults"))):
        raise SweepConfigError(f"the starter gets fewer right answers than the engine's defaults: {'; '.join(lines)}")
    _whole_build(model, [(each.cfg, each.state) for each in swept])
    _publishable_build(model, _build_of(cfg, model, checked.state))
    derived = build_name(cfg.models[model].build.model)
    if name is not None and name != derived:
        raise SweepConfigError(f"--name {name!r} is not the build's name {derived!r}")
    name = derived
    if not _NAME.fullmatch(name):
        raise SweepConfigError(f"the build's name {name!r} must match {_NAME.pattern}, as a profile name must")
    if found is not None:
        _started_from(cfg, model, evidence, found)

    try:
        proposal_text = checked.profile.decode("utf-8") if checked.profile is not None else None
    except UnicodeDecodeError:
        proposal_text = None
    if proposal_text is None:
        raise SweepConfigError(f"{_proposal(cfg, model)[0]} cannot be read as the profile the rule proposed")
    text = starter_text(proposal_text, name, dataset)
    searched = found.evidence if found is not None else None if safe_by == "check" else evidence
    record = starter_evidence(cfg, model, name, dataset, state=checked.state, evidence=evidence, text=text,
                              searched=searched)
    # Staged outside the starters folder, under the names they ship as, and checked there: the evidence as the sweep
    # reads it, its binding to the starter's bytes included, and the starter through the engine's own loader. Both
    # checks read the staging folder alone, so the starters folder is not touched until they pass.
    with tempfile.TemporaryDirectory(prefix="mnemiq-starter-") as scratch:
        staged = Path(scratch)
        starter, shipped = staged / f"{name}.toml", staged / f"{name}{EVIDENCE_SUFFIX}"
        starter.write_bytes(text.encode("utf-8"))
        shipped.write_bytes((record.model_dump_json(indent=2) + "\n").encode("utf-8"))
        read_evidence(name, staged)
        _resolves_as_measured(cfg, model, name, evidence["version"], staged)
        try:
            folder = profiles.starters_folder()
        except profiles.ProfileFileError as exc:
            raise SweepConfigError(str(exc)) from None
        # Taken only now, so every refusal leaves the starters folder as it was, its lock file included.
        with _sole(folder / PUBLISH_LOCK, SweepConfigError(
                f"{folder} is in use by another publisher; publish again once it has ended")):
            _clear_stale_copies(folder)
            # The evidence first: a new starter is never listed without it, and a starter the second move never
            # replaced sits beside evidence that records another file, which the engine refuses.
            _move_in(shipped, folder)
            _move_in(starter, folder)
    return folder / f"{name}.toml", folder / f"{name}{EVIDENCE_SUFFIX}"
