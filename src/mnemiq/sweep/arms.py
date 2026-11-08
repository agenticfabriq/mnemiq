"""The arms of a sweep: what each round runs, and the profile file each arm loads.

An arm is one model in one mode, changed from that model's start. A change is named as a profile names the setting: a
profile-wide setting by its own name (`retrieval_k`), a mode's setting by the mode and its name (`instant.candidates`),
since one profile holds a section for each mode.

The start is the engine's field defaults (each mode's own values, for a mode's settings), overlaid by the start profile
when the config names one. A model's endpoint is a fact about the model, stated once under `[models]` (the judge's under
`[judge]` when the config has one), so it replaces whatever the start file says. Nothing a deployment's env file sets is
part of an arm: `baseline` resets the deployment's settings before the arm's profile applies, and the profile file
writes out every setting that has a value.

Whether an arm can run is the engine's to say. `engine_refusal` loads the arm's own file with the engine's profile
loader, so there is no second table of what combines with what.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from mnemiq.agent.modes import MODES, modes_for
from mnemiq.config import ModeSettings, Settings
from mnemiq.profiles import PROFILE_SETTINGS, load_profiles
from mnemiq.profiles import _read as _read_profile_file
from mnemiq.sweep.config import (
    ENDPOINT_FACTS, SweepConfig, _deployment_ids, deployment_settings, scrub_credentials, start_profile,
)

ROUNDS = ("screen", "grid", "timing", "confirm", "check")
_MODE_FIELDS = tuple(ModeSettings.model_fields)


@dataclass(frozen=True)
class Arm:
    """One profile to run: `model` in `mode`, changed from its start by `changes` (sorted by name).

    `label` is for people, `profile_id` for the engine. An assembled arm runs in each mode of `modes`; its `mode` is the
    first, and `dataclasses.replace(arm, mode=...)` is the same profile in another of them, since neither the round nor
    the mode an arm runs in is part of its `profile_id`.
    """

    model: str
    mode: str
    round: str
    changes: tuple[tuple[str, object], ...]
    modes: tuple[str, ...]
    label: str
    profile_id: str


def _mode_of(name: str) -> str | None:
    """The mode a change belongs to, or None for a profile-wide setting."""
    mode, dot, _ = name.partition(".")
    return mode if dot else None


def _check_name(name: str, mode: str) -> None:
    """Refuse a change no profile of `mode` can carry."""
    of_mode, dot, field = name.partition(".")
    if not dot:
        if name in ENDPOINT_FACTS:
            raise ValueError(f"{name} is where a model is served, stated in [models]; it is not a change")
        if name not in PROFILE_SETTINGS:
            raise ValueError(f"{name} is not a setting a profile can change")
    elif of_mode not in MODES or field not in ModeSettings.model_fields:
        raise ValueError(f"{name} is not a mode's setting")
    elif of_mode != mode:
        raise ValueError(f"{name} is a setting of {of_mode}, and this arm runs in {mode}")


def _show(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "unset" if value is None else str(value)


def _once(values: Iterable[object]) -> list[object]:
    """The values once each, in order."""
    kept: list[object] = []
    for value in values:
        if value not in kept:
            kept.append(value)
    return kept


def _arm(model: str, mode: str, round: str, changes: Mapping[str, object], *, modes: Iterable[str] | None = None,
         assembled: bool = False) -> Arm:
    ordered = tuple(sorted(changes.items(), key=lambda change: change[0]))
    modes = (mode,) if modes is None else tuple(modes)
    label = " + ".join(f"{name}={_show(value)}" for name, value in ordered)
    if not ordered and not assembled:
        return Arm(model, mode, round, ordered, modes, "start", "start")
    body = json.dumps([model, list(modes), [list(change) for change in ordered]], separators=(",", ":"))
    digest = hashlib.sha256(body.encode()).hexdigest()[:10]
    return Arm(model, mode, round, ordered, modes, label or "the start's settings",
               f"{'assembled' if assembled else 'arm'}-{digest}")


def arm_of(model: str, mode: str, changes: Mapping[str, object], *, round: str = "screen") -> Arm:
    """The arm that changes `changes` from the start of `model` in `mode`. No change is the start itself."""
    if mode not in MODES:
        raise ValueError(f"{mode!r} is not a mode")
    if round not in ROUNDS:
        raise ValueError(f"{round!r} is not a round; the rounds are {', '.join(ROUNDS)}")
    for name in changes:
        _check_name(name, mode)
    return _arm(model, mode, round, changes)


def _field_defaults() -> dict[str, object]:
    values = {name: Settings.model_fields[name].get_default(call_default_factory=True)
              for name in sorted(PROFILE_SETTINGS)}
    for mode_name, mode in MODES.items():
        values.update({f"{mode_name}.{field}": getattr(mode, field) for field in _MODE_FIELDS})
    return values


def _endpoint(cfg: SweepConfig, model: str) -> dict[str, object]:
    """Where `model` is served and judged, from the sweep's config."""
    entry = cfg.models[model]
    judge = cfg.judge if cfg.judge is not None else entry
    return {name: getattr(judge if name.startswith("verify_") else entry, name) for name in ENDPOINT_FACTS}


def start_settings(cfg: SweepConfig, model: str) -> dict[str, object]:
    """Every profile setting and every mode's setting `model` starts from, by change name: the engine's field default,
    overlaid by the start profile's entry, with the model's endpoint from `[models]`."""
    values = _field_defaults()
    path, name = start_profile(cfg, model)
    if path is not None:
        read = _read_profile_file(path)
        entry = read.profiles[name or read.default]
        values.update({setting: getattr(entry, setting) for setting in entry.model_fields_set & PROFILE_SETTINGS})
        for mode_name in MODES:
            section = getattr(entry, mode_name)
            if section is not None:
                values.update({f"{mode_name}.{field}": getattr(section, field) for field in section.model_fields_set})
    return values | _endpoint(cfg, model)


def _space(cfg: SweepConfig, mode: str) -> dict[str, list[object]]:
    """The values to try in `mode`, by change name. A mode's setting stated outside a mode's table is tried in every
    mode, beside any values that mode's own table lists."""
    tried: dict[str, list[object]] = {}
    for key, values in cfg.space.items():
        if isinstance(values, dict):
            if key == mode:
                for field, field_values in values.items():
                    tried[f"{mode}.{field}"] = _once([*tried.get(f"{mode}.{field}", []), *field_values])
        else:
            name = key if key in PROFILE_SETTINGS else f"{mode}.{key}"
            tried[name] = _once([*tried.get(name, []), *values])
    return tried


def screen_arms(cfg: SweepConfig) -> list[Arm]:
    """For each model and mode, the start and then one arm for each value to try other than the start's own, by
    setting and then in the order the space lists the values."""
    arms: list[Arm] = []
    for model in cfg.models:
        start = start_settings(cfg, model)
        for mode in cfg.modes:
            arms.append(_arm(model, mode, "screen", {}))
            for name, values in sorted(_space(cfg, mode).items()):
                arms += [_arm(model, mode, "screen", {name: value}) for value in values if value != start[name]]
    return arms


def grid_arms(cfg: SweepConfig, kept: Mapping[tuple[str, str], Mapping[str, Iterable[object]]]) -> list[Arm]:
    """Every combination of the values the screen kept, for each model and mode in `kept`, each setting also left at
    the start's value. A combination of one change is the screen's own arm, returned as it is; only combinations of two
    or more are new, in round `"grid"`. The start itself is not a grid arm."""
    arms: list[Arm] = []
    for (model, mode), settings in kept.items():
        if model not in cfg.models or mode not in cfg.modes:
            raise ValueError(f"the screen kept values for {model} in {mode}, which this sweep does not run")
        start = start_settings(cfg, model)
        options: list[list[tuple[str, object] | None]] = []
        for name in sorted(settings):
            _check_name(name, mode)
            values = [value for value in _once(settings[name]) if value != start[name]]
            if values:
                options.append([None, *((name, value) for value in values)])
        for combination in itertools.product(*options):
            changes = dict(choice for choice in combination if choice is not None)
            if changes:
                arms.append(_arm(model, mode, "screen" if len(changes) == 1 else "grid", changes))
    return arms


def check_arms(cfg: SweepConfig) -> list[Arm]:
    """For each model and mode, the start and then the engine's field defaults (`profile_id` `"defaults"`), whose
    changes undo whatever the start profile set."""
    arms: list[Arm] = []
    for model in cfg.models:
        start = start_settings(cfg, model)
        defaults = _field_defaults() | _endpoint(cfg, model)
        for mode in cfg.modes:
            undo = tuple(sorted(((name, value) for name, value in defaults.items()
                                 if _mode_of(name) in (None, mode) and value != start[name]), key=lambda c: c[0]))
            arms.append(_arm(model, mode, "check", {}))
            arms.append(Arm(model, mode, "check", undo, (mode,), "defaults", "defaults"))
    return arms


def assembled_arm(cfg: SweepConfig, model: str, picks: Mapping[str, Arm],
                  modes: Iterable[str] | None = None) -> tuple[Arm, list[str]]:
    """One profile from each swept mode's pick, to be measured whole in each of those modes: `modes`, in the config's
    order, by default every mode the sweep runs.

    A profile holds one value of a profile-wide setting, so a setting the picks disagree on (one mode's pick changed it
    and another's did not, or changed it otherwise) keeps the start's value. Returned beside the arm, sorted.
    """
    wanted = set(cfg.modes if modes is None else modes)
    modes = tuple(mode for mode in cfg.modes if mode in wanted)
    if not modes or wanted - set(modes):
        raise ValueError(f"an assembled profile spans modes the sweep runs ({', '.join(cfg.modes)}), not "
                         f"{', '.join(sorted(wanted)) or 'none'}")
    if set(picks) != set(modes):
        raise ValueError(f"an assembled profile takes one pick for each of its modes ({', '.join(modes)}), "
                         f"not for {', '.join(sorted(picks)) or 'none'}")
    start = start_settings(cfg, model)
    changes: dict[str, object] = {}
    for mode in modes:
        pick = picks[mode]
        if pick.model != model or pick.mode != mode:
            raise ValueError(f"the pick for {model} in {mode} is an arm of {pick.model} in {pick.mode}")
        for name, value in pick.changes:
            of_mode = _mode_of(name)
            if of_mode is not None and of_mode != mode:
                raise ValueError(f"the pick for {mode} changes {name}, a setting of {of_mode}")
            if of_mode is not None:
                changes[name] = value
    conflicts: list[str] = []
    wide = sorted({name for pick in picks.values() for name, _ in pick.changes if _mode_of(name) is None})
    for name in wide:
        values = [dict(picks[mode].changes).get(name, start[name]) for mode in modes]
        if any(value != values[0] for value in values[1:]):
            conflicts.append(name)
        elif values[0] != start[name]:
            changes[name] = values[0]
    return _arm(model, modes[0], "confirm", changes, modes=modes, assembled=True), conflicts


def _toml(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # TOML's basic string: a backslash, a quote and every control character but none other must be escaped.
        return '"' + "".join("\\\\" if c == "\\" else '\\"' if c == '"'
                             else f"\\u{ord(c):04x}" if ord(c) < 0x20 or ord(c) == 0x7F else c for c in value) + '"'
    raise TypeError(f"a profile file has no way to write {value!r}")


def profile_text(cfg: SweepConfig, arm: Arm) -> str:
    """A profiles file holding one profile, the default, named `arm.profile_id`.

    It writes out every profile setting that has a value. One that is unset cannot be written, since TOML has no null,
    so the arm's settings are `baseline`'s, where each is already unset. Then come the model's key variable, the judge's
    when it has one, and a section for each mode in `arm.modes` with every value that mode's settings hold.
    """
    values = start_settings(cfg, arm.model) | dict(arm.changes)
    entry = cfg.models[arm.model]
    judge_key = (cfg.judge if cfg.judge is not None else entry).verify_api_key_env
    profile = _toml(arm.profile_id)
    lines = [f"default = {profile}", "", f"[profiles.{profile}]", f"api_key_env = {_toml(entry.api_key_env)}"]
    if judge_key:
        lines.append(f"verify_api_key_env = {_toml(judge_key)}")
    lines += [f"{name} = {_toml(values[name])}" for name in sorted(PROFILE_SETTINGS) if values[name] is not None]
    for mode in arm.modes:
        lines += ["", f"[profiles.{profile}.{mode}]"]
        lines += [f"{field} = {_toml(values[f'{mode}.{field}'])}" for field in _MODE_FIELDS
                  if values[f"{mode}.{field}"] is not None]
    return "\n".join(lines) + "\n"


def arm_from_profile(cfg: SweepConfig, model: str, round: str, modes: tuple[str, ...], profile_id: str,
                     path: str | os.PathLike) -> Arm:
    """The arm whose profiles file `profile_text` wrote at `path`, read back by the engine's own reader.

    A change is each setting the file holds at a value other than the start's, where a setting the file leaves out is
    unset, and only the sections of `modes` are read. The arm is rebuilt from those changes as its round built it, and
    refused when it is not `profile_id`: the file was written under another config or another start.
    """
    read = _read_profile_file(os.fspath(path))
    entry = read.profiles[read.default]
    held = {name: getattr(entry, name) if name in entry.model_fields_set else None for name in PROFILE_SETTINGS}
    for mode in modes:
        section = getattr(entry, mode) or ModeSettings()
        held.update({f"{mode}.{field}": getattr(section, field) if field in section.model_fields_set else None
                     for field in _MODE_FIELDS})
    start = start_settings(cfg, model)
    changes = {name: value for name, value in held.items() if value != start[name]}
    if profile_id == "defaults":
        arm = next((replace(arm, round=round) for arm in check_arms(cfg)
                    if arm.model == model and arm.modes == modes and arm.profile_id == "defaults"), None)
    else:
        arm = _arm(model, modes[0], round, changes, modes=modes, assembled=profile_id.startswith("assembled-"))
    if arm is None or arm.profile_id != profile_id or dict(arm.changes) != changes:
        found = arm.profile_id if arm is not None and dict(arm.changes) == changes else "other changes"
        raise ValueError(f"the profile file {path} holds {found!r}, not {profile_id!r}: it was written under another "
                         "config or another start")
    return arm


def baseline(settings: Settings) -> Settings:
    """A deployment's settings as an arm starts from them, before its profile applies: every profile setting at the
    engine's field default and no mode settings of the deployment's own, so nothing its env file sets becomes an unseen
    part of the arm.

    The embedder stays where the deployment had it, since the store's index was built with it. It falls back to the chat
    endpoint, a profile setting, so it is pinned before the chat endpoint is cleared.
    """
    embed_url, embed_key = settings.embed_endpoint()
    reset = {name: Settings.model_fields[name].get_default(call_default_factory=True) for name in PROFILE_SETTINGS}
    return settings.model_copy(update=reset | {"mode_settings": None, "embed_base_url": embed_url,
                                               "embed_api_key": embed_key})


def baselines(cfg: SweepConfig) -> list[tuple[str | None, Settings]]:
    """Each deployment the sweep reads, by db_id (None for one env file), at its `baseline`."""
    return [(db_id, baseline(deployment_settings(cfg, db_id))) for db_id in _deployment_ids(cfg)]


def refusal_under(cfg: SweepConfig, arm: Arm, deployments: list[tuple[str | None, Settings]]) -> str | None:
    """`engine_refusal` against `deployments`, from `baselines`, read once for many arms."""
    with tempfile.TemporaryDirectory(prefix="mnemiq-sweep-arm-") as folder:
        path = os.path.join(folder, "profiles.toml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(profile_text(cfg, arm))
        for db_id, settings in deployments:
            try:
                profiles, default = load_profiles(settings.model_copy(update={"profiles_path": path}))
                modes_for(profiles[default].settings)
            except ValueError as exc:  # the profile loader's refusals, and an unknown mode's
                # A refusal may name a deployment's setting by its value, and a sweep keeps and shows refusals.
                why = scrub_credentials(str(exc), settings)
                return f"the deployment for {db_id!r}: {why}" if db_id is not None else why
    return None


def engine_refusal(cfg: SweepConfig, arm: Arm) -> str | None:
    """Why the engine would refuse to run `arm`, in its own words, or None when it runs.

    The arm's profile file is loaded by the engine's profile loader over each deployment's `baseline`, as a run loads
    it, and its modes are built. A key variable that is not set is named; its value never is, and a deployment URL the
    message quotes is given without its credentials.
    """
    return refusal_under(cfg, arm, baselines(cfg))
