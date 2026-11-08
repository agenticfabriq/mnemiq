"""The profiles a deployment loads: each a model and its settings, per mode.

The best settings depend on the dataset and the model, so they live in a file the deployment owns, and the engine ships
only its defaults. A deployment is one dataset, so a profile is a model and what suits it there.

A profile changes only what `PROFILE_SETTINGS` names. Allow-listed rather than deny-listed: a setting nobody classified
must not become reachable by accident, and `test_every_settings_field_is_classified` makes each new one a decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass
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
    "openwebui_jwt_max_lifetime_s", "openwebui_user_headers", "openwebui_connection_key_header",
    "openwebui_connection_key", "identity_map_path", "mcp_transport",
})

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


def _resolved(name: str, entry: _Entry, base: Settings) -> Settings:
    """`base` under this profile, with each key kept to the endpoint it was set for.

    Judged on the resolved endpoint and not on the fields: `verify_endpoint` falls back to the chat endpoint and the chat
    key, so a cleared field can still send a key to a host it was never meant for.
    """
    update = {field: getattr(entry, field) for field in entry.model_fields_set & PROFILE_SETTINGS}
    sections = {mode: getattr(entry, mode) for mode in MODES if getattr(entry, mode) is not None}
    if sections:
        update["mode_settings"] = sections
    chat_moved = update.get("llm_base_url", base.llm_base_url) != base.llm_base_url
    if entry.api_key_env:
        update["llm_api_key"] = _key_from(entry.api_key_env, name)
    elif chat_moved:
        update["llm_api_key"] = None
    if chat_moved and "llm_context_window" not in update:
        update["llm_context_window"] = None  # the old server's, and a wrong one trims silently
    resolved = base.model_copy(update=update)
    if not resolved.llm_base_url or not resolved.llm_api_key:
        # `LLMClient` refuses both, in words that point at the environment and not at this profile.
        raise ProfileFileError(f"profile {name!r} has no chat endpoint and key: set llm_base_url and api_key_env (a "
                               "server that takes no key still needs the variable to hold some value)")

    judge = resolved.verify_base_url or resolved.llm_base_url
    if entry.verify_api_key_env:
        resolved = resolved.model_copy(update={"verify_api_key": _key_from(entry.verify_api_key_env, name)})
    elif judge != base.verify_endpoint()[0]:
        resolved = resolved.model_copy(update={"verify_api_key": None})
    if judge != base.verify_endpoint()[0] and "verify_model" not in entry.model_fields_set:
        # The old server's. The new one does not serve it, and under fail-closed every judged answer would fail.
        resolved = resolved.model_copy(update={"verify_model": None})
    if judge != resolved.llm_base_url and not resolved.verify_api_key:
        raise ProfileFileError(f"profile {name!r} runs its judge on another endpoint with no key of its own, so it "
                               "would send the chat key there: set verify_api_key_env")
    return resolved


def _refuse_an_unapplied_temperature(name: str, settings: Settings) -> None:
    """`LLMClient` drops a temperature for a model that takes none. It would otherwise boot, get a version and log a value
    the model never sampled at, which is the one thing a version exists to prevent."""
    if settings.writer_temperature is not None and not accepts_temperature(settings.llm_model or ""):
        raise ProfileFileError(f"profile {name!r} sets writer_temperature on {settings.llm_model!r}, which takes none")


def _refuse_reasoning_under_guided_decoding(name: str, settings: Settings) -> None:
    """Guided decoding holds the writer's reply to the JSON from its first token, so the reasoning it asks for could
    never be written: the profile would log a setting that changed nothing but the prompt."""
    if settings.guided_sql and settings.writer_reasoning != "none":
        raise ProfileFileError(f"profile {name!r} sets writer_reasoning with guided_sql, which allows no text before the "
                               "JSON: turn one of them off")


def _checked(name: str, entry: _Entry, default: str, base: Settings) -> Profile:
    if not _NAME.fullmatch(name):
        raise ProfileFileError(f"profile name {name!r} must match {_NAME.pattern}")
    if name == default and entry.roles is not None:
        raise ProfileFileError(f"profile {name!r} is the default, which serves whoever names no profile: "
                               "it takes no roles")
    if name != default and (not entry.roles or not all(role.strip() for role in entry.roles)):
        raise ProfileFileError(f"profile {name!r} must list the roles that may pick it, [\"*\"] meaning anyone")
    settings = _resolved(name, entry, base)
    _refuse_an_unapplied_temperature(name, settings)
    _refuse_reasoning_under_guided_decoding(name, settings)
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
    return {name: _checked(name, entry, read.default, settings) for name, entry in read.profiles.items()}, read.default


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
