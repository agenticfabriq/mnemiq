"""A sweep's config file: one dataset, the models to tune on it, and the values to try for each setting.

The engine is the authority on settings. Each value to try is checked against the engine's own field, its type and its
limits, so a typo or an out-of-range value is refused before any model is called. Only what a profile may change can
vary (`PROFILE_SETTINGS`, and each mode's `ModeSettings`). A model's endpoint is a fact about that model, stated once
under `[models]`, and never a value to try. A key never enters the file: it names the environment variable holding one.

Paths in the file are read from the file's own directory. Paths inside a deployment's env file are read as the engine
reads them, from the working directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator, BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, ValidationError, ValidationInfo,
    field_validator, model_validator,
)

from mnemiq.agent.modes import MODES
from mnemiq.config import (
    _SECRET_HINTS, ModeSettings, Settings, SourceSpec, userinfo_unplaced, without_credentials,
)
from mnemiq.contract import EvaluationCase
from mnemiq.profiles import (
    _NAME, DEPLOYMENT_SETTINGS, ENDPOINT_SETTINGS, PROFILE_SETTINGS, ProfileFileError, _Entry, _starter_entry,
    flattened_entry, profile_sources, profile_starter, starter_path,
)
from mnemiq.profiles import _read as _read_profile_file

# Profile settings that describe where a model is served, not how it is asked: the engine's own list, which a starter
# never sets. They belong to `[models]`, and are refused from `[space]` before the profile-setting check, which would
# otherwise let them through.
ENDPOINT_FACTS = ENDPOINT_SETTINGS
# The names `[start]` reads as its own, so no model may take one.
_START_KEYS = ("profile_file", "profile", "starter")
# A model as the Hugging Face hub names it, `<owner>/<name>`: one slash, and no leading `/`, `.` or `~`, so no absolute,
# relative or home path has this form.
REPO_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
_REPO_ID = re.compile(REPO_ID_PATTERN)
# What a sweep keeps in its out folder beside each model's own folder, `<out>/<model>`, so no model may take one of these
# names: the folder a smoke replaces whole, the proposal's folder, the sweep's record, its split and the cases its
# endpoints left to ask again.
SMOKE = "smoke"
PROPOSAL = "proposal"
SWEEP = "sweep.json"
SPLIT = "split.json"
RETRIES = "retries.json"
OUT_ENTRIES = (SMOKE, PROPOSAL, SWEEP, SPLIT, RETRIES)
# Settings fields whose value is a secret and is left out of the fingerprint. A DSN or URL is kept with its credentials
# taken out (`without_credentials`), since the rest of it names what is read.
_KEY_HINTS = tuple(hint for hint in _SECRET_HINTS if hint != "dsn")
# Sources the engine reads from a local file, which can be hashed. A live database cannot be.
_FILE_SOURCES = ("sqlite", "duckdb")
# How much time and money a sweep may spend, and how long a question is expected to take: none changes an answer,
# so a sweep may be resumed under new ones.
_PLANNING = ("budget_hours", "budget_dollars", "seconds_per_question")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The schemes an endpoint in `[models]` or `[judge]` may use: each is an OpenAI-compatible server.
_WEB_SCHEMES = ("http", "https")
_AT_AFTER_QUERY_OR_FRAGMENT = re.compile(r"[?#][^@]*@")


class SweepConfigError(ValueError):
    """The config cannot be run as written. Nothing runs, and no model is called."""


def _engine_type(model: type[BaseModel], name: str) -> Any:
    """A field's type as the engine declares it, with its limits."""
    field = model.model_fields[name]
    return Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation


def _profile_name(value: str) -> str:
    if not _NAME.fullmatch(value):
        raise ValueError(f"{value!r} must match {_NAME.pattern}, as a profile name must")
    return value


def _env_name(value: str) -> str:
    # The value is not quoted back: a key pasted here by mistake must not reach a log.
    if not _ENV_NAME.fullmatch(value):
        raise ValueError("must name an environment variable, such as MODEL_API_KEY; a key itself never goes in this file")
    return value


def _endpoint_url(value: str | None, info: ValidationInfo) -> str | None:
    """An endpoint as `[models]` and `[judge]` state it: an http or https URL with a host, no credential and no
    fragment. Each arm's profile file and the fingerprint hold it as written. It is never quoted back, since a
    credential is the likeliest thing wrong with it."""
    if value is None:
        return None
    key_env = "verify_api_key_env" if info.field_name.startswith("verify_") else "api_key_env"
    # Checked before any parse, which can stop at a password first: a user or password wherever the engine places one,
    # a credential in the query, an `@` no reading places, or any `@` after a `?` or `#`. A token holding a raw `?` or
    # `#` reads as a host followed by a query or fragment holding the `@`, and a base URL needs no `@` in either.
    if without_credentials(value) != value or userinfo_unplaced(value) or _AT_AFTER_QUERY_OR_FRAGMENT.search(value):
        raise ValueError("carries a credential (a user or password before an '@', or a key in its query), or an '@' "
                         "after a '?' or '#', which a password holding either leaves. A key never goes in this file: "
                         f"name the environment variable that holds it in {key_env}")
    if "#" in value:
        raise ValueError("holds a fragment (a '#' and what follows it), which a base URL never needs, and which each "
                         "arm's profile file and the evidence would hold as written. A key never goes in this file: "
                         f"name the environment variable that holds it in {key_env}")
    try:
        parts = urlsplit(value)
        host, _ = parts.hostname, parts.port  # `port` raises for one that is not a number
    except ValueError:
        raise ValueError("is not a URL that can be read") from None
    if parts.scheme not in _WEB_SCHEMES or not host:
        raise ValueError("must be an http or https URL with a host, such as http://192.0.2.10:8000/v1")
    return value


def _path(value: str | None, info: ValidationInfo) -> str | None:
    """Absolute, read from the config file's directory when relative."""
    if value is None:
        return None
    base = (info.context or {}).get("base") or os.getcwd()
    return os.path.abspath(os.path.join(base, os.path.expanduser(value)))


_Text = Annotated[str, Field(min_length=1)]
# A model's repository id, in the hub's form, so a build read on its own refuses anything else. A sweep config refuses
# one first where the model's name is known, so the refusal names its table, and also one that exists as a path from
# the config's folder (`SweepConfig._models_state_repositories`, `_models_name_repositories`).
_RepoId = Annotated[str, Field(pattern=REPO_ID_PATTERN)]
_ProfileName = Annotated[str, AfterValidator(_profile_name)]
_EnvName = Annotated[str, AfterValidator(_env_name)]
_Count = Annotated[int, Field(ge=0)]
_Positive = Annotated[float, Field(gt=0)]
_Price = Annotated[float, Field(ge=0)]
_Alpha = Annotated[float, Field(gt=0, lt=1)]
_ContextWindow = _engine_type(Settings, "llm_context_window")
_VerifyModel = _engine_type(Settings, "verify_model")
_VerifyBaseUrl = _engine_type(Settings, "verify_base_url")


class _Section(BaseModel):
    # Strict: a count written as `true`, `1.5` or `"2"` is a mistake to refuse, not a value to coerce.
    model_config = ConfigDict(extra="forbid", strict=True)


class _NoKeys(_Section):
    @model_validator(mode="before")
    @classmethod
    def _no_key_in_the_file(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for name in ("api_key", "llm_api_key", "verify_api_key"):
                if name in data:
                    raise ValueError(f"{name}: a key never goes in this file; name the environment variable that "
                                     "holds it in api_key_env or verify_api_key_env")
        return data


class SweepSection(_Section):
    name: _Text
    out: str
    max_errors: _Count = 0
    repeats: Literal["auto"] | Annotated[int, Field(ge=1)] = "auto"

    _out = field_validator("out")(_path)


class Identity(_Section):
    """Who the sweep asks as."""

    tenant: _Text = "local"
    principal: _Text = "local"
    roles: Annotated[list[_Text], Field(min_length=1)]


class Dataset(_Section):
    questions: str
    env: str | None = None  # one deployment's env file
    deployments: dict[_Text, str] | None = None  # db_id to env file
    held_out: Annotated[float, Field(ge=0, lt=1)] = 0.3
    split_seed: int
    identity: Identity

    _paths = field_validator("questions", "env")(_path)

    @field_validator("deployments")
    @classmethod
    def _deployment_paths(cls, value: dict[str, str] | None, info: ValidationInfo) -> dict[str, str] | None:
        return None if value is None else {db_id: _path(env, info) for db_id, env in value.items()}

    @model_validator(mode="after")
    def _one_way_to_name_the_deployment(self) -> Dataset:
        if (self.env is None) == (self.deployments is None):
            raise ValueError("[dataset] names its deployment once: env for one env file, or deployments for an env "
                             "file per db_id")
        if self.deployments is not None and not self.deployments:
            raise ValueError("[dataset] deployments is empty")
        return self


class ModelBuild(_Section):
    """What a model is served from that no server reports reliably, as `[models.<name>.build]` states it: the weights'
    repository id, the GPU and the quantization. Each is optional to a sweep. The server's own account of its build is
    read from it when the sweep runs (`mnemiq.sweep.starters.server_readings`)."""

    model: _RepoId | None = None
    gpu: _Text | None = None
    quantization: _Text | None = None


class ModelEntry(_NoKeys):
    """One model under test: where it is served, the variable holding its key, and the build it is served from."""

    llm_model: _Text
    llm_base_url: _Text
    api_key_env: _EnvName
    llm_context_window: _ContextWindow = None
    verify_model: _VerifyModel = None
    verify_base_url: _VerifyBaseUrl = None
    verify_api_key_env: _EnvName | None = None
    price_in_per_mtok: _Price | None = None
    price_out_per_mtok: _Price | None = None
    build: ModelBuild | None = None

    _endpoints = field_validator("llm_base_url", "verify_base_url")(_endpoint_url)


class Judge(_NoKeys):
    """One judge for every model, replacing each model's own."""

    verify_model: _VerifyModel = None
    verify_base_url: _VerifyBaseUrl = None
    verify_api_key_env: _EnvName | None = None

    _endpoint = field_validator("verify_base_url")(_endpoint_url)

    @model_validator(mode="after")
    def _names_a_judge(self) -> Judge:
        if self.verify_model is None and self.verify_base_url is None:
            raise ValueError("[judge] names no judge: set verify_model, verify_base_url or both")
        return self


class StartEntry(_Section):
    """A start: a profile in a profiles file, or a starter the engine ships, picked by its name."""

    profile_file: str | None = None
    profile: _ProfileName | None = None
    starter: _ProfileName | None = None

    _profile_file = field_validator("profile_file")(_path)

    @model_validator(mode="after")
    def _one_kind(self) -> StartEntry:
        if self.starter is not None and (self.profile_file is not None or self.profile is not None):
            raise ValueError("a start is a starter or a profile file, not both")
        return self


class Start(StartEntry):
    """Where each model starts: `[start]`, or a model's own `[start.<model>]`. Neither set means the engine's defaults."""

    per_model: dict[str, StartEntry] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _model_tables(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        own = {key: value for key, value in data.items() if key in _START_KEYS}
        return {**own, "per_model": {key: value for key, value in data.items() if key not in _START_KEYS}}

    @model_validator(mode="after")
    def _no_profile_beside_a_starter(self) -> Start:
        # A model's own table naming only a profile keeps `[start]`'s file, and a starter's file holds its one profile.
        for model, own in self.per_model.items():
            if self.starter is not None and own.profile is not None and own.profile_file is None:
                raise ValueError(f"[start.{model}] names a profile, and [start] names a starter, which holds no other")
        return self


class Search(_Section):
    strategy: Literal["screen-then-grid", "grid", "check"] = "screen-then-grid"
    screen_alpha: _Alpha = 0.25
    budget_hours: _Positive | None = None
    budget_dollars: _Positive | None = None
    seconds_per_question: _Positive | None = None


# How check mode reads its two arms' wrong and made-up answers against the bounds: run k of the start against run k of
# the defaults, in every run, or each arm's highest count over its runs against the other's.
Harm = Literal["per_run", "worst"]


class Propose(_Section):
    alpha: _Alpha = 0.05
    max_more_wrong: _Count = 0
    max_more_made_up: _Count = 0
    harm: Harm = "per_run"
    profile: _ProfileName | None = None


class Machine(_Section):
    parallel: Annotated[int, Field(ge=1)] = 1


class Control(_Section):
    """Another system's answers to the same questions, read beside the sweep's own."""

    name: _Text
    answers: str

    _answers = field_validator("answers")(_path)


def _tried(where: str, name: str, values: Any, *, in_mode: bool) -> list:
    """The values to try for one setting, as the engine's field types them. The first refusal wins, in this order."""
    if name in ENDPOINT_FACTS:
        raise ValueError(f"{where}: {name} is where a model is served, not a value to try: state it for each model "
                         "in [models]")
    if name in DEPLOYMENT_SETTINGS:
        raise ValueError(f"{where}: {name} cannot vary, since a profile cannot change a deployment setting")
    if in_mode:
        if name not in ModeSettings.model_fields:
            kind = "a profile setting: vary it in [space]" if name in PROFILE_SETTINGS else "not a mode setting"
            raise ValueError(f"{where}: {name} is {kind}")
        declared_by: type[BaseModel] = ModeSettings
    elif name in PROFILE_SETTINGS:
        declared_by = Settings
    elif name in ModeSettings.model_fields:
        declared_by = ModeSettings
    elif name in MODES:
        raise ValueError(f"{where} names a mode: give it a table of that mode's settings")
    else:
        raise ValueError(f"{where}: {name} is not a setting a profile can change")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{where} must be a non-empty list of the values to try")
    adapter = TypeAdapter(_engine_type(declared_by, name))
    typed: list = []
    for value in values:
        try:
            checked = adapter.validate_python(value, strict=True)
        except ValidationError as exc:
            reason = "; ".join(e["msg"] for e in exc.errors(include_input=False, include_url=False))
            raise ValueError(f"{where}: {value!r} is not a value {name} takes: {reason}") from None
        if checked in typed:
            raise ValueError(f"{where} lists {checked!r} twice")
        typed.append(checked)
    return typed


class SweepConfig(BaseModel):
    """A sweep, as its file states it. Each field is one section of the file."""

    model_config = ConfigDict(extra="forbid", strict=True)

    sweep: SweepSection
    dataset: Dataset
    models: Annotated[dict[_ProfileName, ModelEntry], Field(min_length=1)]
    judge: Judge | None = None
    start: Start = Field(default_factory=Start)
    modes: Annotated[list[str], Field(min_length=1)]
    # Setting name to the values to try; a mode's name to a table of that mode's settings. A mode setting stated here
    # rather than under a mode is tried in every mode the sweep runs.
    space: dict[str, Any] = Field(default_factory=dict)
    search: Search = Field(default_factory=Search)
    propose: Propose = Field(default_factory=Propose)
    machine: Machine = Field(default_factory=Machine)
    control: list[Control] = Field(default_factory=list)

    _path: Path | None = PrivateAttr(default=None)

    @property
    def path(self) -> Path | None:
        """The file this config was loaded from."""
        return self._path

    @field_validator("models", mode="before")
    @classmethod
    def _models_state_repositories(cls, models: Any) -> Any:
        """Each build's `model` is a repository id, never a local path: a server's launch argument may be either, and a
        path names nothing on another machine. Refused here, before `ModelBuild` refuses it in pydantic's words, so the
        refusal names the model's table. Anything but text is left to the fields' own types."""
        for name, entry in (models.items() if isinstance(models, dict) else ()):
            build = entry.get("build") if isinstance(entry, dict) else None
            stated = build.get("model") if isinstance(build, dict) else None
            if isinstance(stated, str) and not _REPO_ID.fullmatch(stated):
                raise ValueError(f"[models.{name}.build] model {stated!r} is not a repository id, <owner>/<name>: "
                                 "state the id the weights were downloaded as")
        return models

    @field_validator("models")
    @classmethod
    def _models_name_repositories(cls, models: dict[str, ModelEntry], info: ValidationInfo) -> dict[str, ModelEntry]:
        """A relative path such as `models/qwen-awq` has a repository id's form, so a build's `model` is also refused
        when it exists as a path from the config's folder."""
        base = (info.context or {}).get("base") or os.getcwd()
        for name, entry in models.items():
            stated = entry.build.model if entry.build is not None else None
            if stated is not None and os.path.lexists(os.path.join(base, stated)):
                raise ValueError(f"[models.{name}.build] model {stated!r} is a path here, not a repository id: state "
                                 "the id the weights were downloaded as")
        return models

    @field_validator("modes")
    @classmethod
    def _known_modes(cls, modes: list[str]) -> list[str]:
        for mode in modes:
            if mode not in MODES:
                raise ValueError(f"{mode!r} is not a mode; the modes are {sorted(MODES)}")
        if len(set(modes)) != len(modes):
            raise ValueError("a mode is listed twice")
        return modes

    @model_validator(mode="after")
    def _consistent(self) -> SweepConfig:
        space: dict[str, Any] = {}
        for key, values in self.space.items():
            if key in MODES and isinstance(values, dict):
                space[key] = {name: _tried(f"space.{key}.{name}", name, tried, in_mode=True)
                              for name, tried in values.items()}
            else:
                space[key] = _tried(f"space.{key}", key, values, in_mode=False)
        self.space = space
        for mode in (key for key, values in space.items() if isinstance(values, dict)):
            if mode not in self.modes:
                raise ValueError(f"[space.{mode}] varies a mode that [modes] does not run")
        for name in self.models:
            if name in _START_KEYS:
                raise ValueError(f"models.{name}: [start] reads {name} as its own key, so no model can take the name")
            if name in OUT_ENTRIES:
                raise ValueError(f"models.{name}: the sweep keeps its own {name} in its out folder, beside each model's "
                                 "folder, so no model can take the name")
        for name in self.start.per_model:
            if name not in self.models:
                raise ValueError(f"[start.{name}] names no model in [models]")
        names = [control.name for control in self.control]
        if len(set(names)) != len(names):
            raise ValueError("[[control]] names one control twice")
        if self.propose.harm == "worst" and self.search.strategy != "check":
            raise ValueError('[propose] harm = "worst" reads check mode\'s start and defaults, each at its highest '
                             f"count over its runs, and strategy {self.search.strategy!r} compares no such pair: use "
                             'it with [search] strategy = "check", or leave it out')
        if self.propose.harm == "worst" and not (isinstance(self.sweep.repeats, int) and self.sweep.repeats >= 2):
            raise ValueError(f'[propose] harm = "worst" reads each arm at its highest count over its runs, and [sweep] '
                             f"repeats is {self.sweep.repeats!r}: with one run of each arm it is run against run, and "
                             '"auto" runs every arm once when the start writes the same SQL twice, as a start at '
                             "temperature 0 does. Set [sweep] repeats to 2 or more")
        if self.search.strategy != "check" and self.dataset.held_out == 0:
            raise ValueError(f"[dataset] held_out is 0, and a search (strategy {self.search.strategy!r}) confirms its "
                             "pick on held-out questions: set held_out above 0, or use check mode ([search] strategy "
                             '= "check"), which holds none out')
        return self


def _refusal(where: Path | str, exc: ValidationError) -> SweepConfigError:
    """Where and why only, never the value: pydantic's own message quotes the input, and a key pasted into the file is the
    likeliest mistake to quote."""
    reasons = []
    for error in exc.errors(include_input=False, include_url=False, include_context=False):
        at = ".".join(str(part) for part in error["loc"])
        why = error["msg"].removeprefix("Value error, ")
        reasons.append(f"{at}: {why}" if at else why)
    return SweepConfigError(f"{where}: {'; '.join(reasons)}")


def load_config(path: str | os.PathLike) -> SweepConfig:
    """The sweep config at `path`, checked against the engine's fields and against the files it names."""
    path = Path(path).absolute()
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except OSError as exc:
        raise SweepConfigError(f"{path} cannot be read: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise SweepConfigError(f"{path} is not valid TOML: {exc}") from None
    try:
        cfg = SweepConfig.model_validate(raw, context={"base": str(path.parent)})
    except ValidationError as exc:
        # Not chained: the chained error's text would carry the input this message leaves out.
        raise _refusal(path, exc) from None
    cfg._path = path
    for db_id in _deployment_ids(cfg):
        deployment_settings(cfg, db_id)
    cases = load_questions(cfg)
    case_ids = {case.id for case in cases}
    if cfg.search.strategy != "check":
        from mnemiq.sweep.split import split  # the split reads this module

        if not split(cfg, cases)["held_out"]:
            raise SweepConfigError(f"{path}: [dataset] held_out {cfg.dataset.held_out:g} holds no question of the "
                                   f"{len(cases)} the question file has (it takes held_out of each group, those with "
                                   "an answer and those without, rounded down), and a search confirms its pick on "
                                   "held-out questions: raise held_out or add questions")
    for model in cfg.models:
        _check_start(cfg, model)
    if cfg.search.strategy == "check":
        _check_differs(cfg)
    for control in cfg.control:
        control_answers(control, case_ids)
    return cfg


def control_answers(control: Control, case_ids: Collection[str]) -> list[dict]:
    """A control's answers, refused unless its file is a non-empty JSON list in which each answer is `{id, sql}`, with
    SQL, or `{id, declined: true}`, and names a case of `case_ids` once. `load_config` reads them, so a control that
    cannot be graded is refused before any arm runs."""
    where = f"[[control]] {control.name!r}: its answers {control.answers}"
    try:
        raw = json.loads(Path(control.answers).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SweepConfigError(f"{where} cannot be read: {exc.strerror}") from None
    except ValueError:
        raise SweepConfigError(f"{where} are not JSON") from None
    if not isinstance(raw, list) or not raw:
        raise SweepConfigError(f"{where} must be a non-empty JSON list")
    seen: set[str] = set()
    for number, answer in enumerate(raw, start=1):
        if not (isinstance(answer, dict) and isinstance(answer.get("id"), str) and (
                (answer.keys() == {"id", "sql"} and isinstance(answer["sql"], str) and answer["sql"].strip())
                or (answer.keys() == {"id", "declined"} and answer["declined"] is True))):
            raise SweepConfigError(f"{where}: answer {number} is neither {{id, sql}} nor {{id, declined: true}}")
        if answer["id"] not in case_ids:
            raise SweepConfigError(f"{where}: answer {number} names case {answer['id']!r}, which the question file "
                                   "does not hold")
        if answer["id"] in seen:
            raise SweepConfigError(f"{where}: case {answer['id']!r} is answered twice")
        seen.add(answer["id"])
    return raw


def _deployment_ids(cfg: SweepConfig) -> list[str | None]:
    return [None] if cfg.dataset.deployments is None else sorted(cfg.dataset.deployments)


def _env_file(cfg: SweepConfig, db_id: str | None) -> str:
    if cfg.dataset.deployments is None:
        return cfg.dataset.env  # one deployment answers every question
    if db_id is None:
        raise SweepConfigError("[dataset] names a deployment for each db_id, so a question must say which it reads")
    if db_id not in cfg.dataset.deployments:
        raise SweepConfigError(f"[dataset] deployments has no db_id {db_id!r}")
    return cfg.dataset.deployments[db_id]


class _EnvFileOnly(Settings):
    """Settings read from an env file alone, never from the environment of the process reading it."""

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings):
        return (init_settings, dotenv_settings)


def deployment_settings(cfg: SweepConfig, db_id: str | None = None) -> Settings:
    """The settings of the deployment `db_id` reads, from its env file and nothing else.

    `MNEMIQ_*` variables in this process's environment are ignored: a shell or a test may set them, and a sweep that read
    them would measure a deployment other than the one its file names.
    """
    path = _env_file(cfg, db_id)
    if not os.path.isfile(path):
        raise SweepConfigError(f"the env file {path} cannot be read")
    try:
        read = _EnvFileOnly(_env_file=path, _env_file_encoding="utf-8")
    except ValidationError as exc:
        raise _refusal(path, exc) from None
    # Copied into a plain `Settings`, unvalidated since `read` was validated: `Settings.model_validate` would read the
    # environment again.
    return Settings.model_construct(_fields_set=set(read.model_fields_set),
                                    **{name: getattr(read, name) for name in Settings.model_fields})


def _source_specs(settings: Settings) -> list[SourceSpec]:
    try:
        return settings.source_specs()
    except OSError as exc:
        raise SweepConfigError(f"the source manifest {settings.sources_path} cannot be read: {exc.strerror}") from None
    except (ValueError, KeyError, TypeError) as exc:
        raise SweepConfigError(f"the source manifest {settings.sources_path} cannot be read: {exc}") from None


def load_questions(cfg: SweepConfig) -> list[EvaluationCase]:
    """The cases in the question file, refused unless every one can be graded: a unique id, a question, gold SQL for an
    answerable case, and a db_id that names a deployment (or, with one env file, nothing or one of its sources)."""
    path = cfg.dataset.questions
    try:
        raw = json.loads(Path(path).read_text())
    except OSError as exc:
        raise SweepConfigError(f"the question file {path} cannot be read: {exc.strerror}") from None
    except ValueError:
        raise SweepConfigError(f"the question file {path} is not JSON") from None
    if not isinstance(raw, list) or not raw:
        raise SweepConfigError(f"the question file {path} must be a non-empty JSON list of cases")
    cases: list[EvaluationCase] = []
    for number, item in enumerate(raw, start=1):
        try:
            cases.append(EvaluationCase.model_validate(item))
        except ValidationError as exc:
            raise _refusal(f"{path}, case {number}", exc) from None
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise SweepConfigError(f"{path}: case id {case.id!r} appears twice")
        seen.add(case.id)
        if not case.question.strip():
            raise SweepConfigError(f"{path}: case {case.id!r} has no question")
        if case.answerable and not (case.gold_sql or "").strip():
            raise SweepConfigError(f"{path}: case {case.id!r} is answerable and has no gold_sql to grade it by")
    if cfg.dataset.deployments is not None:
        known, where = set(cfg.dataset.deployments), "[dataset] deployments"
    elif any(case.db_id is not None for case in cases):
        known = {spec.id for spec in _source_specs(deployment_settings(cfg, None))}
        where = "the deployment's sources"
    else:
        known, where = set(), ""
    for case in cases:
        if case.db_id is None and cfg.dataset.deployments is None:
            continue
        if case.db_id not in known:
            raise SweepConfigError(f"{path}: case {case.id!r} names db_id {case.db_id!r}, which is not one of {where} "
                                   f"({', '.join(sorted(known))})")
    return cases


def _named_starter(cfg: SweepConfig, model: str) -> str | None:
    """The starter `model`'s start names, by name: its own table's when that names a starter or a file, and so replaces
    `[start]` whole, else `[start]`'s. Never matched from the model's name."""
    own = cfg.start.per_model.get(model)
    if own is not None and (own.starter is not None or own.profile_file is not None):
        return own.starter
    return cfg.start.starter


def start_profile(cfg: SweepConfig, model: str) -> tuple[str | None, str | None]:
    """The profile file and profile `model` starts from; (None, None) is the engine's defaults. A starter is its file
    and the one profile it holds, named as the starter is. A model's own table replaces `[start]`: naming a starter or
    a file, it replaces it whole; naming only a profile, it keeps `[start]`'s file."""
    if (starter := _named_starter(cfg, model)) is not None:
        try:
            return starter_path(starter), starter
        except ProfileFileError as exc:
            raise SweepConfigError(f"model {model!r} starts from a starter the engine refuses: {exc}") from None
    own = cfg.start.per_model.get(model)
    if own is not None and own.profile_file is not None:
        return own.profile_file, own.profile
    profile = own.profile if own is not None and own.profile is not None else cfg.start.profile
    return cfg.start.profile_file, profile


def _start_named(cfg: SweepConfig, model: str) -> tuple[str, str] | None:
    """The file `model` starts from and the profile in it, the file's default when the start names none; None for the
    engine's defaults."""
    path, profile = start_profile(cfg, model)
    if path is None:
        return None
    return path, profile or _read_profile_file(path).default


def start_entry(cfg: SweepConfig, model: str) -> _Entry | None:
    """What `model`'s start sets, read whole: each layer of its profile's chain merged (`flattened_entry`), the starter
    its root names first, then each profile it extends, then its own settings. None for the engine's defaults."""
    named = _start_named(cfg, model)
    return None if named is None else flattened_entry(*named)


def start_starter(cfg: SweepConfig, model: str) -> str | None:
    """The starter `model`'s start resolves through: one its start names, or the one its start profile's root names
    (`profile_starter`). None when it reaches none."""
    if (starter := _named_starter(cfg, model)) is not None:
        return starter
    named = _start_named(cfg, model)
    return None if named is None else profile_starter(*named)


def unset_keys(cfg: SweepConfig, models: Collection[str]) -> list[str]:
    """Each key variable `models` name, their judge's included, that this process's environment does not set, after
    the model naming it. Named, never quoted: the engine's profile loader refuses a profile whose variable is unset."""
    unset = []
    for model in models:
        entry = cfg.models[model]
        judge = cfg.judge if cfg.judge is not None else entry
        unset += [f"{variable} (model {model!r})" for variable in (entry.api_key_env, judge.verify_api_key_env)
                  if variable and not os.environ.get(variable)]
    return unset


def _check_differs(cfg: SweepConfig) -> None:
    """Refuse a check-mode sweep whose start, for some model and mode, is the engine's defaults: check mode compares the
    two, so it would compare two identical profiles."""
    from mnemiq.sweep.arms import check_arms  # the arms read this module

    for arm in check_arms(cfg):
        if arm.profile_id == "defaults" and not arm.changes:
            own = start_profile(cfg, arm.model)[0] is not None
            where = (f"the start profile of model {arm.model!r} sets nothing in {arm.mode} that the engine's defaults "
                     "do not" if own else f"model {arm.model!r} starts from the engine's defaults")
            raise SweepConfigError(f"{cfg.path}: {where}, and check mode compares the start with those defaults, so it "
                                   "would compare two identical profiles. Start from a profile that differs from them "
                                   f"([start] or [start.{arm.model}]), or search ([search] strategy).")


def _check_start(cfg: SweepConfig, model: str) -> None:
    """Refuse a start the sweep cannot read whole: a file that cannot be read, a profile it does not hold, a chain the
    engine refuses (a cycle, a missing parent, a starter that sets what a starter never sets), or a starter whose
    evidence `read_evidence` refuses, since the sweep compares builds against it."""
    from mnemiq.sweep.starters import read_evidence  # the starters read this module

    named = _named_starter(cfg, model)
    path, profile = start_profile(cfg, model)
    if path is None:
        if profile is not None:
            raise SweepConfigError(f"model {model!r} starts from profile {profile!r} but no profile_file holds it")
        return
    if not os.path.isfile(path):
        raise SweepConfigError(f"model {model!r} starts from {path}, which cannot be read")
    try:
        read = _read_profile_file(path)
    except ProfileFileError as exc:
        raise SweepConfigError(f"model {model!r} starts from a profile file the engine refuses: {exc}") from None
    name = profile or read.default
    if name not in read.profiles:
        raise SweepConfigError(f"model {model!r} starts from profile {name!r}, which {path} does not hold "
                               f"(it holds {sorted(read.profiles)})")
    what = f"starter {named!r}" if named is not None else f"profile {name!r}"
    try:
        if named is not None:
            _starter_entry(named)  # the engine's own reading of a starter, which refuses what a starter never sets
        flattened_entry(path, name)
        starter = start_starter(cfg, model)
    except ProfileFileError as exc:
        raise SweepConfigError(f"model {model!r} starts from {what}, which the engine refuses: {exc}") from None
    if starter is not None:
        try:
            read_evidence(starter)
        except SweepConfigError as exc:
            raise SweepConfigError(f"model {model!r} starts from {what}: {exc}") from None


def _sha256(path: str, what: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()
    except OSError as exc:
        raise SweepConfigError(f"{what} {path} cannot be read: {exc.strerror}") from None


def _start_files(cfg: SweepConfig, model: str) -> dict[str, str]:
    """Each file `model`'s start is read from, by path, to what it is: its profile file and the starter its chain
    reaches (`profile_sources`), and that starter's evidence, which the sweep reads to compare builds."""
    from mnemiq.sweep.starters import evidence_path  # the starters read this module

    try:
        named = _start_named(cfg, model)
        if named is None:
            return {}
        starter = start_starter(cfg, model)
        shipped = starter_path(starter) if starter is not None else None
        files = {path: "the start's starter" if path == shipped else "the start profile file"
                 for path in profile_sources(*named)}
        if starter is not None:
            files[str(evidence_path(starter))] = "the start's starter evidence"
    except ProfileFileError as exc:
        # Read again as each arm runs: a refusal here marks the arm invalid, where the engine's error would end it.
        raise SweepConfigError(f"the start of model {model!r} cannot be read: {exc}") from None
    return files


def pinned_files(cfg: SweepConfig) -> dict[str, str]:
    """Each file an answer depends on, by absolute path, to its sha256: each model's start profile, the starter its
    start reaches and that starter's evidence, the question file, and for each deployment its access policy, its store
    and each source read from a file. A live database is not here: it cannot be hashed."""
    files = {cfg.dataset.questions: "the question file"}
    for model in cfg.models:
        files.update(_start_files(cfg, model))
    for db_id in _deployment_ids(cfg):
        settings = deployment_settings(cfg, db_id)
        if settings.authz_path:
            files[os.path.abspath(settings.authz_path)] = "the access policy"
        files[os.path.abspath(settings.store_path)] = "the store"
        for spec in _source_specs(settings):
            if spec.kind in _FILE_SOURCES:
                files[os.path.abspath(spec.target)] = f"the source {spec.id!r}"
    return {path: _sha256(path, files[path]) for path in sorted(files)}


def _holds_a_url(name: str) -> bool:
    return name.endswith(("_dsn", "_url"))


def scrub_credentials(text: str, settings: Settings) -> str:
    """`text` with each DSN or URL of `settings` that carries a credential in its `without_credentials` form, as written
    and as a Python repr writes it, for a message that may name a setting by its value."""
    carried = {value for name in Settings.model_fields if _holds_a_url(name)
               and isinstance(value := getattr(settings, name), str) and without_credentials(value) != value}
    for value in sorted(carried, key=len, reverse=True):
        clean = without_credentials(value)
        text = text.replace(value, clean).replace(repr(value)[1:-1], clean)
    return text


def _answering_settings(settings: Settings) -> dict:
    """A deployment's settings as the fingerprint holds them: no key, and no credential in a DSN or URL."""
    view: dict[str, Any] = {}
    for name, value in settings.model_dump(mode="json").items():
        if any(hint in name for hint in _KEY_HINTS):
            continue
        if isinstance(value, str) and _holds_a_url(name):
            value = without_credentials(value)
        view[name] = value
    view["sources"] = [asdict(spec) | {"target": without_credentials(spec.target)} for spec in _source_specs(settings)]
    return view


def fingerprint_body(cfg: SweepConfig) -> dict:
    """What `fingerprint` hashes. A key a config gained with starters and build tables (`[start] starter`, a model's
    own `starter`, `[models.<name>.build]`) enters only when set, and `[propose] harm` only when it is "worst", so a
    config that names none of them fingerprints as it did before them."""
    config = cfg.model_dump(mode="json")
    config["search"] = {key: value for key, value in config["search"].items() if key not in _PLANNING}
    del config["control"]
    if config["propose"]["harm"] == "per_run":
        del config["propose"]["harm"]
    for table in (config["start"], *config["start"]["per_model"].values()):
        if table.get("starter") is None:
            table.pop("starter", None)
    for entry in config["models"].values():
        if entry.get("build") is None:
            entry.pop("build", None)
    return {
        "config": config,
        "deployments": {db_id or "": _answering_settings(deployment_settings(cfg, db_id))
                        for db_id in _deployment_ids(cfg)},
        "files": pinned_files(cfg),
        "split_seed": cfg.dataset.split_seed,
    }


def fingerprint(cfg: SweepConfig) -> str:
    """What a sweep's answers depend on, as one sha256, so that a sweep resumed under anything else is refused.

    It covers the config, less `[search]`'s budgets and `seconds_per_question` and less `[[control]]` (a sweep may be
    resumed with a larger budget or a new estimate, or given a control afterwards), each model's build table among it;
    each deployment's settings without its keys or the credentials in its DSNs and URLs; the pinned files, a start's
    starter and its evidence among them; and the split seed. The config's own endpoints carry no credential:
    `load_config` refuses one that does.
    """
    body = fingerprint_body(cfg)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
