"""The profiles a deployment loads: each a model and its settings, per mode.

The best settings depend on the dataset and the model, so they live in a file the deployment owns, and the engine ships
only its defaults. A deployment is one dataset, so a profile is a model and what suits it there.

A profile changes only what `PROFILE_SETTINGS` names. Allow-listed rather than deny-listed: a setting nobody classified
must not become reachable by accident, and `test_every_settings_field_is_classified` makes each new one a decision.

A profile can have one parent: another profile in its file (`extends`) or a starter the engine ships (`starter`). Its
layers resolve in order, the deployment first, each over the one before it by the rules a profile with no parent
follows over the deployment.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace as _Sample

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from mnemiq.agent import synthesize
from mnemiq.agent.loop import STRATEGIES
from mnemiq.agent.modes import MODES, modes_for
from mnemiq.config import ModeSettings, Settings, userinfo_unplaced, without_credentials
from mnemiq.execute import select
from mnemiq.execute.select import ClusterView
from mnemiq.generate import correct, prompts
from mnemiq.llm.client import accepts_temperature
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.verify import judge

# What changes how a question is answered, and nothing that decides who may see what.
PROFILE_SETTINGS = frozenset({
    "llm_base_url", "llm_model", "llm_seed", "llm_context_window", "llm_prompt_cut_check", "llm_window_check",
    "card_style", "guided_sql", "assertive_sql", "retrieval_k", "guard_undefined_terms", "guard_fanout",
    "answer_markdown", "verify_threshold", "verify_sanity", "verify_grounding", "verify_base_url", "verify_model",
    "writer_temperature", "prompt_layout", "writer_reasoning", "writer_rules",
})

DEPLOYMENT_SETTINGS = frozenset({
    # Who may see what, and where the data goes.
    "local_only", "authz_path", "write_enabled", "verify_fail_closed", "verify_override", "principal", "roles", "tenant",
    "verity_records_url", "verity_watermark_path", "verity_page_size", "verity_full_resync_after_secs",
    "verity_traces_url", "verity_trace_send_text", "verity_trace_send_identity_detail", "verity_trace_send_rows",
    "verity_token_url", "verity_client_id", "verity_client_secret",
    # The dataset: its source, its store and what built the store. The embedder is tied to the store's index.
    "pg_dsn", "oracle_user", "oracle_password", "oracle_config_dir", "oracle_wallet_password", "oracle_pool_max",
    "oracle_acquire_timeout_s", "oracle_probe_timeout_s", "oracle_read_only_ttl_s", "sources_path", "source_id",
    "store_path", "control_dsn", "embed_model", "embed_base_url", "embed_api_key", "enrich_facts", "enrich_examples",
    "dictionary_path", "ontology_records_path", "definition_index_max_concepts", "binding_suggestions_path",
    # Secrets are never read from a file: a profile names a variable instead.
    "llm_api_key", "verify_api_key",
    # The deployment's own running.
    "profiles_path", "mode_settings", "default_mode", "max_question_chars", "ack_advisories", "verify_judge", "acme_data_dir",
    "answer_log_path", "answer_log_text", "engine_commit",
    # Who is asking: each request's identity from a verified token.
    "oidc_issuer", "oidc_audience", "oidc_jwks_url", "oidc_jwks_file", "oidc_tenant_claim", "oidc_roles_claim",
    "oidc_groups_claim", "oidc_leeway_s", "openwebui_audience", "openwebui_jwt_secret", "openwebui_jwt_header",
    "openwebui_jwt_max_lifetime_s", "openwebui_user_headers", "openwebui_connection_key_header", "openwebui_task_header",
    "openwebui_connection_key", "identity_map_path", "mcp_transport",
})

# The profile settings that belong to an endpoint, the deployment's own: a starter never sets them.
ENDPOINT_SETTINGS: tuple[str, ...] = ("llm_model", "llm_base_url", "llm_context_window", "verify_model", "verify_base_url")

# The starters this engine ships: one TOML file per model build, inside the package, so an installed engine reads them
# offline.
STARTER_PACKAGE = "mnemiq.starter_profiles"

# What a starter is refused for setting: a deployment's endpoint and keys, the roles its file grants, and a parent.
_NOT_IN_A_STARTER = frozenset({*ENDPOINT_SETTINGS, "api_key_env", "verify_api_key_env", "roles", "extends", "starter"})

# What one layer passes to the layers above it, besides its mode sections. Roles, the description and the parent keys
# stay with the profile that names them, as does being the file's default.
_INHERITED = PROFILE_SETTINGS | {"api_key_env", "verify_api_key_env"}

_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")


class ProfileFileError(ValueError):
    """The profile file cannot be run as written. Boot stops: an arm whose setting never ran reads as "no effect"."""


@dataclass(frozen=True)
class Profile:
    name: str
    roles: frozenset[str] | None  # None: the default, for whoever names no profile
    settings: Settings


class UnknownProfile(ValueError):
    """A request named a profile it may not pick. Unknown and forbidden read alike, so a name cannot disclose that a
    profile exists. `offered` is what this caller may pick, and is all a door may tell them."""

    def __init__(self, offered: list[str]) -> None:
        super().__init__("that profile is not one this deployment offers you")
        self.offered = offered


def may_pick(profile: Profile, identity) -> bool:
    """Read as `FileAuthzProvider.grants_for` reads an identity: roles and groups alike."""
    if profile.roles is None or "*" in profile.roles:
        return True
    return bool(profile.roles & {*identity.roles, *identity.groups})


def offered_to(identity, profiles) -> list[str]:
    return sorted(name for name, profile in profiles.items() if may_pick(profile, identity))


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    roles: list[str] | None = None
    # A profile's parent, at most one of the two: another profile in the file, or a starter the engine ships.
    extends: str | None = None
    starter: str | None = None
    api_key_env: str | None = None
    verify_api_key_env: str | None = None
    instant: ModeSettings | None = None
    thinking: ModeSettings | None = None
    deep: ModeSettings | None = None


# Typed as `Settings` types them, so a value a deployment could not set in its environment cannot be set here.
_ProfileEntry = create_model(
    "_ProfileEntry", __base__=_Entry,
    **{name: (Settings.model_fields[name].annotation, Settings.model_fields[name]) for name in sorted(PROFILE_SETTINGS)})


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str
    profiles: dict[str, _ProfileEntry]


def _unreadable(path: str, exc: ValidationError) -> ProfileFileError:
    """Field and reason only: pydantic's own message quotes the input, and a key pasted into the file is the likeliest
    mistake this catches."""
    errors = exc.errors(include_input=False, include_url=False, include_context=False)
    reasons = "; ".join(f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in errors)
    return ProfileFileError(f"{path}: {reasons}")


def _read(path: str) -> _File:
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except OSError as exc:
        raise ProfileFileError(f"MNEMIQ_PROFILES_PATH={path!r} cannot be read: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        # Position only: the decoder's message can quote the line.
        raise ProfileFileError(f"{path} is not valid TOML") from exc
    try:
        return _File.model_validate(raw)
    except ValidationError as exc:
        raise _unreadable(path, exc) from exc


def _key_from(variable: str, name: str) -> str:
    value = os.environ.get(variable)
    if not value:
        raise ProfileFileError(f"profile {name!r} names the key variable {variable}, which is not set")
    return value


def starters_folder() -> Path:
    """The folder of the starters this engine ships, on disk. Every reader calls it through this module and never binds
    it at import, so a test that points it at a folder of its own reaches them all."""
    try:
        folder = importlib.resources.files(STARTER_PACKAGE)
    except ModuleNotFoundError as exc:
        raise ProfileFileError(f"this engine has no {STARTER_PACKAGE} package, so it ships no starters") from exc
    if not isinstance(folder, Path) or not folder.is_dir():
        raise ProfileFileError(f"{STARTER_PACKAGE} is not a folder on disk, as in a zipped install, so its starters "
                               "cannot be read")
    return folder.resolve()


def starter_names() -> list[str]:
    """The starters this engine ships, by name: the stem of each TOML file in its starters folder."""
    return sorted(path.stem for path in starters_folder().glob("*.toml") if path.is_file())


def starter_path(name: str) -> str:
    """The file of a starter this engine ships. Only a name the folder lists is taken, so a name shaped like a path
    never reaches a file outside it, and one an upgrade removed is refused rather than read as no starter."""
    names = starter_names()
    if name not in names:
        raise ProfileFileError(f"starter {name!r} is not one this engine ships; it ships: {', '.join(names) or 'none'}")
    return str(starters_folder() / f"{name}.toml")


def _starter_entry(name: str) -> _Entry:
    """A starter's settings: its file's default profile. Answer-time settings and mode sections only, since the model,
    its endpoints, the keys and who may pick a profile stay with the deployment, and a starter is always a chain's first
    layer."""
    read = _read(starter_path(name))
    if read.default not in read.profiles:
        raise ProfileFileError(f"starter {name!r}: default = {read.default!r} names no profile")
    entry = read.profiles[read.default]
    if owned := sorted(entry.model_fields_set & _NOT_IN_A_STARTER):
        raise ProfileFileError(f"starter {name!r} sets {', '.join(owned)}, which a starter never sets: the model, its "
                               "endpoints, the keys and the roles stay with the deployment, and a starter has no parent")
    return entry


def _lineage(read: _File, name: str) -> tuple[str | None, list[str]]:
    """The starter `name`'s root names, if any, and the profiles from the root down to `name`, following `extends`."""
    if name not in read.profiles:
        raise ProfileFileError(f"profile {name!r} is not one this file holds; it holds {sorted(read.profiles)}")
    chain = [name]
    while True:
        holder = chain[-1]
        entry = read.profiles[holder]
        if entry.extends is not None and entry.starter is not None:
            raise ProfileFileError(f"profile {holder!r} names two parents, extends {entry.extends!r} and starter "
                                   f"{entry.starter!r}: a profile has at most one")
        if entry.extends is None:
            return entry.starter, chain[::-1]
        if entry.extends in chain:
            raise ProfileFileError(f"profile {name!r} extends a chain that comes back to itself: "
                                   + " -> ".join([*chain, entry.extends]))
        if entry.extends not in read.profiles:
            raise ProfileFileError(f"profile {holder!r} extends {entry.extends!r}, which this file does not hold")
        chain.append(entry.extends)


def _merged_sections(below: dict[str, ModeSettings], entry: _Entry) -> dict[str, ModeSettings]:
    """`entry`'s mode sections over `below`, setting by setting. Only what was set is carried, so `modes_for` still
    applies only that, and a section with nothing below it is taken as it is."""
    merged = dict(below)
    for mode in MODES:
        if (section := getattr(entry, mode)) is None:
            continue
        merged[mode] = section if mode not in merged else ModeSettings(**{
            field: getattr(layer, field) for layer in (merged[mode], section) for field in layer.model_fields_set})
    return merged


def _layered(name: str, entry: _Entry, base: Settings) -> Settings:
    """`base` under one layer of a profile, with each key kept to the endpoint it was set for. `base` is the layer
    below: the deployment, a starter or a parent. So a layer moves an endpoint when it differs from the one below it,
    even when it points back at the deployment's own.

    Judged on the resolved endpoint and not on the fields: `verify_endpoint` falls back to the chat endpoint and the chat
    key, so a cleared field can still send a key to a host it was never meant for. Whether the result has a chat key, and
    a judge key where it needs one, is checked once every layer is applied (`_checked`), since a layer below the one that
    names the endpoint need not have one.
    """
    update = {field: getattr(entry, field) for field in entry.model_fields_set & PROFILE_SETTINGS}
    # With no parent, `base` is the deployment, whose `mode_settings` is None beside a profiles file (`load_profiles`
    # refuses both), so the profile's sections are taken as they are.
    if sections := _merged_sections(base.mode_settings or {}, entry):
        update["mode_settings"] = sections
    chat_moved = update.get("llm_base_url", base.llm_base_url) != base.llm_base_url
    if entry.api_key_env:
        update["llm_api_key"] = _key_from(entry.api_key_env, name)
    elif chat_moved:
        update["llm_api_key"] = None
    if chat_moved and "llm_context_window" not in update:
        update["llm_context_window"] = None  # the old server's, and a wrong one trims silently
    resolved = base.model_copy(update=update)

    judge = resolved.verify_base_url or resolved.llm_base_url
    if entry.verify_api_key_env:
        resolved = resolved.model_copy(update={"verify_api_key": _key_from(entry.verify_api_key_env, name)})
    elif judge != base.verify_endpoint()[0]:
        resolved = resolved.model_copy(update={"verify_api_key": None})
    if judge != base.verify_endpoint()[0] and "verify_model" not in entry.model_fields_set:
        # The old server's. The new one does not serve it, and under fail-closed every judged answer would fail.
        resolved = resolved.model_copy(update={"verify_model": None})
    return resolved


def _set_by(field: str, layers: list[tuple[str, _Entry]]) -> str:
    """Where `field`'s value came from: the last of `layers` that set it, or the deployment when none did."""
    return next((label for label, entry in reversed(layers) if field in entry.model_fields_set), "the deployment")


def _refuse_an_unapplied_temperature(name: str, settings: Settings,
                                     layers: list[tuple[str, _Entry]] | None = None) -> None:
    """`LLMClient` drops a temperature for a model that takes none. It would otherwise boot, get a version and log a value
    the model never sampled at, which is the one thing a version exists to prevent.

    With a profiles file, `layers` names where the value came from, since a parent's or a starter's is not written in
    the profile being refused."""
    if settings.writer_temperature is not None and not accepts_temperature(settings.llm_model or ""):
        where = "" if layers is None else f" (the value comes from {_set_by('writer_temperature', layers)})"
        raise ProfileFileError(f"profile {name!r} sets writer_temperature on {settings.llm_model!r}, which takes "
                               f"none{where}")


def _refuse_reasoning_under_guided_decoding(name: str, settings: Settings,
                                            layers: list[tuple[str, _Entry]] | None = None) -> None:
    """Guided decoding holds the writer's reply to the JSON from its first token, so the reasoning it asks for could
    never be written: the profile would log a setting that changed nothing but the prompt.

    With a profiles file, `layers` names where each of the two came from."""
    if settings.guided_sql and settings.writer_reasoning != "none":
        where = "" if layers is None else (f" (writer_reasoning from {_set_by('writer_reasoning', layers)}, "
                                           f"guided_sql from {_set_by('guided_sql', layers)})")
        raise ProfileFileError(f"profile {name!r} sets writer_reasoning with guided_sql, which allows no text before the "
                               f"JSON: turn one of them off{where}")


def _checked(name: str, read: _File, base: Settings) -> Profile:
    entry = read.profiles[name]
    if not _NAME.fullmatch(name):
        raise ProfileFileError(f"profile name {name!r} must match {_NAME.pattern}")
    if name == read.default and entry.roles is not None:
        raise ProfileFileError(f"profile {name!r} is the default, which serves whoever names no profile: "
                               "it takes no roles")
    if name != read.default and (not entry.roles or not all(role.strip() for role in entry.roles)):
        raise ProfileFileError(f"profile {name!r} must list the roles that may pick it, [\"*\"] meaning anyone")
    starter, chain = _lineage(read, name)
    # Each layer as it is applied, so a refusal can name the one a value came from.
    layers: list[tuple[str, _Entry]] = []
    settings = base
    if starter is not None:
        layers.append((f"starter {starter!r}", _starter_entry(starter)))
        settings = _layered(starter, layers[-1][1], settings)
    for link in chain:
        layers.append((f"profile {link!r}", read.profiles[link]))
        settings = _layered(link, read.profiles[link], settings)

    if not settings.llm_base_url or not settings.llm_api_key:
        # `LLMClient` refuses both, in words that point at the environment and not at this profile.
        raise ProfileFileError(f"profile {name!r} has no chat endpoint and key: set llm_base_url and api_key_env (a "
                               "server that takes no key still needs the variable to hold some value)")
    if (settings.verify_base_url or settings.llm_base_url) != settings.llm_base_url and not settings.verify_api_key:
        raise ProfileFileError(f"profile {name!r} runs its judge on another endpoint with no key of its own, so it "
                               "would send the chat key there: set verify_api_key_env")
    _refuse_an_unapplied_temperature(name, settings, layers)
    _refuse_reasoning_under_guided_decoding(name, settings, layers)
    try:
        modes_for(settings)
    except ValueError as exc:
        raise ProfileFileError(f"profile {name!r}: {exc}") from exc
    if offenders := settings.local_only_offenders():
        raise ProfileFileError(f"profile {name!r} would leave this network under MNEMIQ_LOCAL_ONLY: "
                               + "; ".join(offenders))
    return Profile(name, None if entry.roles is None else frozenset(entry.roles), settings)


def load_profiles(settings: Settings) -> tuple[dict[str, Profile], str]:
    """The profiles this deployment runs, and the default's name. With no file: one profile, these settings as they are."""
    if "MNEMIQ_PROFILE" in os.environ:
        # `Settings` ignores what it does not know, so this would silently stop meaning what it says.
        raise ProfileFileError("MNEMIQ_PROFILE is gone: name a file of profiles in MNEMIQ_PROFILES_PATH, or unset it")
    if not settings.profiles_path:
        _refuse_an_unapplied_temperature("default", settings)
        _refuse_reasoning_under_guided_decoding("default", settings)
        return {"default": Profile("default", None, settings)}, "default"
    if settings.mode_settings:
        raise ProfileFileError("MNEMIQ_PROFILES_PATH and MNEMIQ_MODE_SETTINGS are both set: "
                               "state each mode's settings in the file")
    embed_url, embed_key = settings.embed_endpoint()
    if not embed_url or not embed_key:
        raise ProfileFileError("the embedder stays on this deployment's settings, since the store's index was built with "
                               "it: set MNEMIQ_EMBED_BASE_URL and MNEMIQ_EMBED_API_KEY, or MNEMIQ_LLM_BASE_URL and "
                               "MNEMIQ_LLM_API_KEY")
    read = _read(settings.profiles_path)
    if read.default not in read.profiles:
        raise ProfileFileError(f"default = {read.default!r} names no profile; the file has {sorted(read.profiles)}")
    return {name: _checked(name, read, settings) for name in read.profiles}, read.default


def flattened_entry(path: str, name: str) -> _Entry:
    """What each layer of `name`'s chain sets, merged: the starter first, then each profile from the root down to `name`,
    mode sections setting by setting. For a reader that needs what a profile sets rather than what a deployment
    resolves it to, such as the start of a sweep, so no endpoint rule is applied: those need a deployment.

    Roles, the description and the parent keys are not carried, so what it reports as set is what the layers set that
    is inherited."""
    read = _read(path)
    starter, chain = _lineage(read, name)
    layers = ([_starter_entry(starter)] if starter is not None else []) + [read.profiles[link] for link in chain]
    values: dict[str, object] = {}
    sections: dict[str, ModeSettings] = {}
    for entry in layers:
        values.update({field: getattr(entry, field) for field in entry.model_fields_set & _INHERITED})
        sections = _merged_sections(sections, entry)
    return _ProfileEntry(**values, **sections)


def profile_starter(path: str, name: str) -> str | None:
    """The starter `name`'s chain starts from: the one its root names, or None."""
    return _lineage(_read(path), name)[0]


def profile_sources(path: str, name: str) -> list[str]:
    """Every file `name` resolves from: its profiles file, then its starter's file when its root names one."""
    starter = profile_starter(path, name)
    return [path] if starter is None else [path, starter_path(starter)]


def _sample_packet() -> ContextPacket:
    """Every section the writer's user prompt can hold, so a change to any of them moves the version. Fixed text: only
    the templates vary. Duck-typed, since `meaning_sections` reads attributes and a real record would add nothing."""
    return ContextPacket(
        question="Q", cards=[RetrievedCard(object_id="t", card="C", score=1.0)], grant_fingerprint="f",
        enrichment_version=None,
        definitions=[_Sample(term="T", definition="D")],
        metrics=[_Sample(label="L", id="i", measure=_Sample(source="s", expr="e"))],
        dimensions=[_Sample(label="L", source="s", expression=lambda: "e")],
        concepts=[_Sample(column_id="c", scheme_label="s", notation="n", label="l")],
        examples=[_Sample(question="q", sql="s")],
        history=[_Sample(question="q", sql="s", columns=["c"], rows=[[1]])],
    )


def _prompt_texts(settings: Settings, dialect: str) -> dict[str, str]:
    """Every prompt a model call uses, as these settings render it. Read through each module at call time."""
    texts = {f"writer system {strategy}": prompts.system_prompt(
        dialect=dialect, strategy=strategy, **prompts.writer_style(settings)) for strategy in STRATEGIES}
    texts["writer user"] = prompts.user_prompt(_sample_packet(), "F")
    if settings.prompt_layout == "question_last":
        for strategy in STRATEGIES:
            texts[f"writer system {strategy}"], texts["writer user"] = prompts.question_last_prompt(
                texts[f"writer system {strategy}"], _sample_packet(), "F")
    texts["corrector system"] = correct._SYSTEM
    texts["corrector user"] = correct.user_text("s", "p")
    texts["judge system"] = judge._SYSTEM
    texts["judge user"] = judge._prompt("q", "s", "x", "p")
    texts["selector system"] = select._SYSTEM
    texts["selector user"] = select._judge_prompt("q", [ClusterView(sql="s", preview="p", size=1)])
    for rows in (1, synthesize.LIST_LIMIT + 1):
        for forced in (False, True):
            texts[f"prose system {rows} {forced}"] = synthesize.system_text(settings.answer_markdown, rows, forced)
    texts["prose user"] = synthesize.user_text("q", "s", "r")
    return texts


def _no_credentials(value):
    """A setting as it may be logged. A URL goes through the engine's one rule for that (`without_credentials`): no user
    or password, and no query parameter named like a credential. A URL whose `@` no reading can place is not logged at
    all. Anything else, a model name with an `@` in it included, is logged as it is."""
    if not isinstance(value, str) or "://" not in value:
        return value
    return "(unreadable url)" if userinfo_unplaced(value) else without_credentials(value)


def resolved_view(settings: Settings, dialect: str) -> dict:
    """What this profile runs, with no secret in it: its answer-time settings, its modes, what the deployment sets that
    changes an answer, and a hash of each prompt."""
    modes = {name: asdict(mode) for name, mode in modes_for(settings).items()}
    if settings.verify_override in {"0", "1"}:  # as `_resolve_verify_level` reads it: what each mode's verifier will do
        for mode in modes.values():
            mode["verify"] = "off" if settings.verify_override == "0" else "full"
    return {
        "dialect": dialect,
        "settings": {name: _no_credentials(getattr(settings, name)) for name in sorted(PROFILE_SETTINGS)},
        "modes": modes,
        # Out of a profile's reach, and they still decide what an answer is.
        "deployment": {"verify_fail_closed": settings.verify_fail_closed, "max_question_chars": settings.max_question_chars},
        "prompts": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in _prompt_texts(settings, dialect).items()},
    }


def profile_version(settings: Settings, dialect: str) -> str:
    """The hash of what this profile resolves to, not of the file: a profile lists only what differs, so an engine
    upgrade could otherwise change what an unchanged file means. Card rendering and the decider are not hashed, so an
    answer stamps the engine's commit beside this."""
    canonical = json.dumps(resolved_view(settings, dialect), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
