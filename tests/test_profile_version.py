"""A profile's version is what it resolves to, so an answer can always say what produced it.

A profile lists only what differs, so an engine upgrade could change what an unchanged file means. The version hashes the
resolved settings, the resolved modes and every prompt a model call uses. Card rendering and the decider are not in it:
the version and the engine commit together pin the engine.
"""

from __future__ import annotations

import json
import os

import pytest

from mnemiq.config import Settings
from mnemiq.profiles import load_profiles, profile_version, resolved_view

LOCAL = "http://localhost:8000/v1"


def _settings(**fields) -> Settings:
    fields.setdefault("llm_base_url", LOCAL)
    fields.setdefault("llm_api_key", "planted-chat-key")
    fields.setdefault("llm_model", "m")
    return Settings(acme_data_dir=None, **fields)


def _from_file(tmp_path, text: str, name: str = "only", **fields) -> Settings:
    path = tmp_path / "profiles.toml"
    path.write_text(text)
    return load_profiles(_settings(profiles_path=str(path), **fields))[0][name].settings


def test_the_same_contents_give_the_same_version(tmp_path):
    one = _from_file(tmp_path, 'default = "only"\n[profiles.only]\nllm_model = "n"\nretrieval_k = 3\n')
    two = _from_file(tmp_path, 'default = "only"\n[profiles.only]\nretrieval_k = 3\nllm_model = "n"\n')
    assert profile_version(one, "duckdb") == profile_version(two, "duckdb")


def test_it_is_short_and_stable_in_form():
    version = profile_version(_settings(), "duckdb")
    assert len(version) == 12 and int(version, 16) >= 0


def test_a_section_changes_it(tmp_path):
    plain = _from_file(tmp_path, 'default = "only"\n[profiles.only]\nllm_model = "n"\n')
    bare = _from_file(tmp_path, 'default = "only"\n[profiles.only]\nllm_model = "n"\n[profiles.only.instant]\ncard_style = "bare"\n')
    assert profile_version(plain, "duckdb") != profile_version(bare, "duckdb")


def test_an_inherited_value_changes_it_with_the_file_untouched(tmp_path):
    text = 'default = "only"\n[profiles.only]\nllm_model = "n"\n'
    assert (profile_version(_from_file(tmp_path, text, retrieval_k=5), "duckdb")
            != profile_version(_from_file(tmp_path, text, retrieval_k=6), "duckdb"))


def test_the_dialect_changes_it():
    assert profile_version(_settings(), "duckdb") != profile_version(_settings(), "oracle")


@pytest.mark.parametrize("field, value", [("verify_override", "0"), ("verify_override", "1"), ("verify_fail_closed", False),
                                          ("max_question_chars", 5)])
def test_what_a_deployment_sets_that_changes_answers_changes_it(field, value):
    """Per deployment, so out of a profile's reach, and it still decides what an answer is. One version on two boots that
    answer differently would say nothing."""
    assert profile_version(_settings(**{field: value}), "duckdb") != profile_version(_settings(), "duckdb")


def test_the_view_says_what_the_verifier_will_do():
    view = resolved_view(_settings(verify_override="0"), "duckdb")
    assert {name: mode["verify"] for name, mode in view["modes"].items()} == {"instant": "off", "thinking": "off", "deep": "off"}
    assert resolved_view(_settings(verify_fail_closed=False), "duckdb")["deployment"]["verify_fail_closed"] is False


def test_who_may_pick_it_does_not_change_it(tmp_path):
    text = 'default = "d"\n[profiles.d]\nllm_model = "n"\n[profiles.only]\nllm_model = "n"\nroles = ["{}"]\n'
    assert (profile_version(_from_file(tmp_path, text.format("a")), "duckdb")
            == profile_version(_from_file(tmp_path, text.format("b")), "duckdb"))


@pytest.mark.parametrize("module, name", [
    ("mnemiq.generate.prompts", "PERSONA"),
    ("mnemiq.generate.prompts", "_DEFER_DEFAULT"),
    ("mnemiq.generate.correct", "_SYSTEM"),
    ("mnemiq.verify.judge", "_SYSTEM"),
    ("mnemiq.execute.select", "_SYSTEM"),
    ("mnemiq.agent.synthesize", "_PERSONA"),
    ("mnemiq.agent.synthesize", "_OPEN_PLAIN"),
    ("mnemiq.agent.synthesize", "_RULES"),
    ("mnemiq.agent.synthesize", "_SUMMARISE"),
    ("mnemiq.agent.synthesize", "_FORCED"),
])
def test_each_prompt_changes_it_with_the_file_untouched(monkeypatch, module, name):
    """An engine upgrade that changes what a model is told must not leave the version where it was."""
    import importlib

    before = profile_version(_settings(), "duckdb")
    target = importlib.import_module(module)
    monkeypatch.setattr(target, name, getattr(target, name) + " changed")
    assert profile_version(_settings(), "duckdb") != before


@pytest.mark.parametrize("module, name", [
    ("mnemiq.generate.prompts", "user_prompt"),
    ("mnemiq.generate.correct", "user_text"),
    ("mnemiq.verify.judge", "_prompt"),
    ("mnemiq.execute.select", "_judge_prompt"),
    ("mnemiq.agent.synthesize", "user_text"),
])
def test_each_user_template_changes_it(monkeypatch, module, name):
    import importlib

    before = profile_version(_settings(), "duckdb")
    target = importlib.import_module(module)
    original = getattr(target, name)
    monkeypatch.setattr(target, name, lambda *a, **k: original(*a, **k) + " changed")
    assert profile_version(_settings(), "duckdb") != before


def test_the_prose_markdown_variant_counts_when_the_profile_uses_it(monkeypatch):
    from mnemiq.agent import synthesize

    before = profile_version(_settings(answer_markdown=True), "duckdb")
    monkeypatch.setattr(synthesize, "_MARKDOWN", synthesize._MARKDOWN + " changed")
    assert profile_version(_settings(answer_markdown=True), "duckdb") != before


def test_the_view_holds_no_secret():
    view = json.dumps(resolved_view(_settings(verify_api_key="planted-judge-key"), "duckdb"))
    assert "planted-chat-key" not in view
    assert "planted-judge-key" not in view


def test_the_view_drops_credentials_written_into_a_url():
    view = json.dumps(resolved_view(_settings(llm_base_url="https://user:planted-url-secret@host.example/v1"), "duckdb"))
    assert "planted-url-secret" not in view
    assert "host.example" in view


def test_the_view_says_what_ran():
    view = resolved_view(_settings(llm_model="n", retrieval_k=9), "duckdb")
    assert view["settings"]["llm_model"] == "n"
    assert view["settings"]["retrieval_k"] == 9
    assert view["modes"]["instant"]["candidates"] == 1
    assert view["dialect"] == "duckdb"


@pytest.mark.parametrize("field, url", [("llm_base_url", "http://h:8000/v1?api_key=planted-query-secret"),
                                        ("verify_base_url", "http://judge:8001/v1?access_token=planted-query-secret")])
def test_the_view_drops_a_credential_in_a_urls_query(field, url):
    view = resolved_view(_settings(**{field: url}), "duckdb")
    assert "planted-query-secret" not in json.dumps(view)
    assert view["settings"][field] == url.split("?")[0]


@pytest.mark.parametrize("url", ["https://user:planted/secret@host.example/v1",    # a raw `/` in the password
                                 "https://user:planted#secret@host.example/v1"])   # a raw `#`, which moves the host
def test_the_view_drops_a_password_whatever_it_holds(url):
    assert "planted" not in json.dumps(resolved_view(_settings(llm_base_url=url), "duckdb"))


@pytest.mark.parametrize("url", ["http://localhost:8000/v1?version=2", "http://host/models@v2",
                                 "https://api.example.com/models/vendor-model@2024/v1", "https://host/v1?user=a@b",
                                 "https://host/v1#frag@x"])
def test_a_setting_with_no_credential_is_logged_as_it_is(monkeypatch, url):
    settings = _settings(llm_base_url=url, llm_model="vendor-model@2024-06-20",
                         verify_base_url="https://Judge.Example.net:8443/v1")
    view, version = resolved_view(settings, "duckdb"), profile_version(settings, "duckdb")
    monkeypatch.setattr("mnemiq.profiles._no_credentials", lambda value: value)
    assert view == resolved_view(settings, "duckdb") and version == profile_version(settings, "duckdb")


# What `profile_version(_settings(writer_temperature=0.0), "duckdb")` gave on main before the temperature floor for a
# candidate that repeats a strategy, with every MNEMIQ_* variable cleared. A change that moves every version on purpose
# records it again on main, and says so in its commit.
VERSION_ON_MAIN = "cec13cfd4071"


def test_the_floor_changes_no_profile_version(monkeypatch):
    """A profile at writer temperature 0, whose version hashes deep's mode with the rest, resolves to the version it
    had on main before the floor: the floor is engine code, not a setting or a mode field."""
    for name in [n for n in os.environ if n.startswith("MNEMIQ_")]:
        monkeypatch.delenv(name)                    # so a shell's setting cannot move the version
    assert profile_version(_settings(writer_temperature=0.0), "duckdb") == VERSION_ON_MAIN


def test_a_mode_field_added_later_moves_a_version_only_where_a_profile_sets_it(monkeypatch):
    """`empty_retries` and `retry_answered_empties` at 0 and off run as the engine ran before they existed, so they
    leave the version as it was (the test above), and a profile that sets either gets a version of its own."""
    for name in [n for n in os.environ if n.startswith("MNEMIQ_")]:
        monkeypatch.delenv(name)
    base = profile_version(_settings(writer_temperature=0.0), "duckdb")
    retry = profile_version(_settings(writer_temperature=0.0, mode_settings={"instant": {"empty_retries": 1}}), "duckdb")
    every = profile_version(_settings(writer_temperature=0.0, mode_settings={
        "instant": {"empty_retries": 1, "retry_answered_empties": True}}), "duckdb")
    assert base == VERSION_ON_MAIN and len({base, retry, every}) == 3
