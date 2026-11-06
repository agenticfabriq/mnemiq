"""A deployment names its profiles in a file: each a model and its settings, per mode.

The best settings depend on the dataset and the model, so they belong to a deployment and not to the engine's code.
A profile reaches only what changes how a question is answered. What decides who may see what stays per deployment.
"""

from __future__ import annotations

import pytest

from mnemiq.agent.modes import MODES, modes_for
from mnemiq.config import Settings
from mnemiq.profiles import (DEPLOYMENT_SETTINGS, PROFILE_SETTINGS, ProfileFileError, load_profiles)

LOCAL = "http://localhost:8000/v1"
HOSTED = "https://models.example.com/v1"
OTHER = "https://judge.example.net/v1"

_TWO = f"""
default = "qwen"

[profiles.qwen]
llm_base_url = "{LOCAL}"
llm_model = "qwen"
[profiles.qwen.instant]
card_style = "bare"

[profiles.gpt]
llm_base_url = "{HOSTED}"
llm_model = "gpt"
api_key_env = "TEST_GPT_KEY"
roles = ["analyst"]
"""


def _settings(**fields) -> Settings:
    fields.setdefault("llm_base_url", LOCAL)
    fields.setdefault("llm_api_key", "local-key")
    fields.setdefault("llm_model", "base-model")
    return Settings(acme_data_dir=None, **fields)


def _load(tmp_path, text: str, **fields):
    path = tmp_path / "profiles.toml"
    path.write_text(text)
    return load_profiles(_settings(profiles_path=str(path), **fields))


def _one(extra: str = "") -> str:
    return f'default = "only"\n[profiles.only]\nllm_model = "m"\n{extra}'


def _other(body: str) -> str:
    return f'default = "only"\n[profiles.only]\nllm_model = "m"\n[profiles.other]\nroles = ["*"]\n{body}'


def test_every_settings_field_is_classified():
    """A new setting has to be placed. Left out, it would silently stay per deployment, or worse, not."""
    assert set(Settings.model_fields) == PROFILE_SETTINGS | DEPLOYMENT_SETTINGS
    assert not PROFILE_SETTINGS & DEPLOYMENT_SETTINGS


@pytest.mark.parametrize("name", ["local_only", "write_enabled", "authz_path", "store_path", "control_dsn",
                                  "embed_model", "embed_base_url", "verify_fail_closed", "verity_traces_url",
                                  "llm_api_key", "verify_api_key", "sources_path", "enrich_facts"])
def test_what_decides_access_is_not_a_profile_setting(name):
    assert name not in PROFILE_SETTINGS
    assert name in DEPLOYMENT_SETTINGS


def test_no_file_gives_one_default_profile():
    settings = _settings()
    profiles, default = load_profiles(settings)
    assert default == "default"
    assert list(profiles) == ["default"]
    assert profiles["default"].settings is settings
    assert profiles["default"].roles is None


def test_a_section_reaches_its_mode_and_only_its_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_GPT_KEY", "gpt-key")
    profiles, default = _load(tmp_path, _TWO)
    assert default == "qwen"
    modes = modes_for(profiles["qwen"].settings)
    assert modes["instant"].card_style == "bare"
    assert modes["thinking"] == MODES["thinking"]
    assert modes_for(profiles["gpt"].settings) == MODES


def test_a_profile_inherits_what_it_does_not_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_GPT_KEY", "gpt-key")
    profiles, _ = _load(tmp_path, _TWO, retrieval_k=7)
    assert profiles["gpt"].settings.retrieval_k == 7
    assert profiles["gpt"].settings.llm_model == "gpt"
    assert profiles["gpt"].roles == frozenset({"analyst"})
    assert profiles["qwen"].roles is None


@pytest.mark.parametrize("text", [
    "not toml [",
    'default = "nobody"\n[profiles.only]\nllm_model = "m"',
    '[profiles.only]\nllm_model = "m"',
    _one('write_enabled = true'),
    _one('llm_api_key = "k"'),
    _one('[profiles.only.fast]\ncard_style = "bare"'),
    _one('retrieval_k = "many"'),
    _one('roles = ["analyst"]'),
    _one('[profiles.only.instant]\njudge = true'),
    _one('[profiles.only.instant]\ncard_style = "fancy"'),
    _other('llm_model = "n"\napi_key_env = "TEST_NOT_SET"'),
    'default = "only"\n[profiles.only]\nllm_model = "m"\n[profiles.other]\nllm_model = "n"',
    'default = "only"\n[profiles.only]\nllm_model = "m"\n[profiles.other]\nroles = []',
    'default = "only"\n[profiles.only]\nllm_model = "m"\n[profiles.other]\nroles = [" "]',
    'default = "Only One"\n[profiles."Only One"]\nllm_model = "m"',
    'default = "only"\nstray = 1\n[profiles.only]\nllm_model = "m"',
])
def test_a_file_that_cannot_be_read_as_written_is_refused(tmp_path, monkeypatch, text):
    monkeypatch.delenv("TEST_NOT_SET", raising=False)
    with pytest.raises(ProfileFileError):
        _load(tmp_path, text)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ProfileFileError, match="nope.toml"):
        load_profiles(_settings(profiles_path=str(tmp_path / "nope.toml")))


def test_a_file_and_inline_mode_settings_are_refused_together(tmp_path):
    with pytest.raises(ProfileFileError, match="MNEMIQ_MODE_SETTINGS"):
        _load(tmp_path, _one(), mode_settings={"instant": {"card_style": "bare"}})


def test_a_leftover_mnemiq_profile_is_refused(monkeypatch):
    """`Settings` ignores unknown variables, so it would silently stop meaning what it says."""
    monkeypatch.setenv("MNEMIQ_PROFILE", "qwen-32b")
    with pytest.raises(ProfileFileError, match="MNEMIQ_PROFILE"):
        load_profiles(_settings())


def test_a_refusal_never_repeats_what_the_file_holds(tmp_path):
    """A key pasted into the file is the likeliest mistake, and pydantic's own message quotes the input."""
    with pytest.raises(ProfileFileError) as refused:
        _load(tmp_path, _one('llm_api_key = "sk-planted-secret"'))
    assert "sk-planted-secret" not in str(refused.value)


def test_a_refusal_names_the_profile(tmp_path):
    with pytest.raises(ProfileFileError, match="only"):
        _load(tmp_path, _one('[profiles.only.instant]\njudge = true'))


def test_a_file_needs_the_embedder_stated(tmp_path):
    """The embedder stays on the deployment's settings, and with a file those may hold no endpoint at all."""
    path = tmp_path / "profiles.toml"
    path.write_text(_one())
    settings = Settings(acme_data_dir=None, llm_base_url=None, llm_api_key=None, embed_base_url=None,
                        embed_api_key=None, profiles_path=str(path))
    with pytest.raises(ProfileFileError, match="MNEMIQ_EMBED_BASE_URL"):
        load_profiles(settings)


# --- a key goes only to the endpoint it was set for -----------------------------------------------------------------


_HOSTED_CHAT = f'llm_base_url = "{HOSTED}"\napi_key_env = "TEST_CHAT_KEY"'


def _moved(tmp_path, monkeypatch, body: str, **fields):
    monkeypatch.setenv("TEST_CHAT_KEY", "chat-env-key")
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-env-key")
    profiles, _ = _load(tmp_path, _other(body), **fields)
    return profiles["other"].settings


def test_a_new_chat_endpoint_with_no_key_is_refused_naming_the_profile(tmp_path, monkeypatch):
    """`LLMClient` refuses an empty key, and a boot crash saying "set MNEMIQ_LLM_*" would point at the wrong fix."""
    with pytest.raises(ProfileFileError, match="other.*api_key_env"):
        _moved(tmp_path, monkeypatch, f'llm_base_url = "{HOSTED}"')


def test_a_new_chat_endpoint_takes_the_key_it_names_and_not_the_old_one(tmp_path, monkeypatch):
    got = _moved(tmp_path, monkeypatch, _HOSTED_CHAT)
    assert got.llm_api_key == "chat-env-key"
    assert got.verify_endpoint() == (HOSTED, "chat-env-key")


def test_a_new_chat_endpoint_drops_the_old_window(tmp_path, monkeypatch):
    assert _moved(tmp_path, monkeypatch, _HOSTED_CHAT, llm_context_window=32768).llm_context_window is None


def test_a_new_chat_endpoint_keeps_a_window_it_states(tmp_path, monkeypatch):
    got = _moved(tmp_path, monkeypatch, _HOSTED_CHAT + "\nllm_context_window = 4096", llm_context_window=32768)
    assert got.llm_context_window == 4096


def test_a_kept_endpoint_keeps_its_key_and_window(tmp_path, monkeypatch):
    got = _moved(tmp_path, monkeypatch, f'llm_base_url = "{LOCAL}"\nllm_model = "n"', llm_context_window=32768)
    assert got.llm_api_key == "local-key"
    assert got.llm_context_window == 32768


def test_a_profile_with_no_chat_endpoint_at_all_is_refused_naming_it(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text(_one())
    settings = Settings(acme_data_dir=None, llm_base_url=None, llm_api_key=None, embed_base_url=LOCAL,
                        embed_api_key="embed-key", profiles_path=str(path))
    with pytest.raises(ProfileFileError, match="only"):
        load_profiles(settings)


def test_the_base_judge_key_does_not_follow_the_chat_to_a_new_host(tmp_path, monkeypatch):
    """With no `verify_base_url` the judge goes wherever the chat goes, and would take `verify_api_key` with it."""
    got = _moved(tmp_path, monkeypatch, _HOSTED_CHAT, verify_api_key="base-judge-key")
    assert got.verify_api_key is None
    assert got.verify_endpoint() == (HOSTED, "chat-env-key")


def test_the_base_judge_model_does_not_follow_the_chat_to_a_new_host(tmp_path, monkeypatch):
    """The new server does not serve it, and under fail-closed every judged answer on this profile would fail."""
    got = _moved(tmp_path, monkeypatch, _HOSTED_CHAT, verify_model="old-servers-judge")
    assert got.verify_model is None


def test_a_judge_model_the_profile_states_is_kept(tmp_path, monkeypatch):
    got = _moved(tmp_path, monkeypatch, _HOSTED_CHAT + '\nverify_model = "new-judge"', verify_model="old-servers-judge")
    assert got.verify_model == "new-judge"


def test_a_judge_that_stays_put_keeps_its_model(tmp_path, monkeypatch):
    got = _moved(tmp_path, monkeypatch, _HOSTED_CHAT, verify_base_url=OTHER, verify_api_key="k", verify_model="its-judge")
    assert got.verify_model == "its-judge"


def test_a_kept_chat_keeps_the_judge_model(tmp_path, monkeypatch):
    assert _moved(tmp_path, monkeypatch, 'llm_model = "n"', verify_model="its-judge").verify_model == "its-judge"


@pytest.mark.parametrize("judge", [OTHER, "http://localhost:8001/v1"])
def test_a_judge_on_another_endpoint_with_no_key_of_its_own_is_refused(tmp_path, monkeypatch, judge):
    """`verify_endpoint` would hand it the chat key. Another port is another server."""
    with pytest.raises(ProfileFileError, match="verify_api_key_env"):
        _moved(tmp_path, monkeypatch, f'verify_base_url = "{judge}"')


def test_a_judge_on_another_host_takes_the_key_it_names(tmp_path, monkeypatch):
    got = _moved(tmp_path, monkeypatch, f'verify_base_url = "{OTHER}"\nverify_api_key_env = "TEST_JUDGE_KEY"')
    assert got.verify_endpoint() == (OTHER, "judge-env-key")
    assert got.llm_api_key == "local-key"


def test_a_judge_that_stays_put_keeps_its_key(tmp_path, monkeypatch):
    got = _moved(tmp_path, monkeypatch, _HOSTED_CHAT, verify_base_url=OTHER, verify_api_key="base-judge-key")
    assert got.verify_endpoint() == (OTHER, "base-judge-key")
    assert got.llm_api_key == "chat-env-key"


# --- local only -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("body", [_HOSTED_CHAT, f'verify_base_url = "{HOSTED}"\nverify_api_key_env = "TEST_JUDGE_KEY"'])
def test_local_only_checks_every_profile(tmp_path, monkeypatch, body):
    monkeypatch.setenv("TEST_CHAT_KEY", "chat-env-key")
    monkeypatch.setenv("TEST_JUDGE_KEY", "judge-env-key")
    with pytest.raises(ProfileFileError, match="other.*leave this network"):
        _load(tmp_path, _other(body), local_only=True)


def test_local_only_passes_a_file_that_stays_local(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CHAT_KEY", "chat-env-key")
    body = 'llm_base_url = "http://10.0.0.5:8000/v1"\napi_key_env = "TEST_CHAT_KEY"'
    profiles, _ = _load(tmp_path, _other(body), local_only=True)
    assert "other" in profiles
