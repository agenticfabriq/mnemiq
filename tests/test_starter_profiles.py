"""A profile can start from a starter the engine ships, picked by name: the starter over the deployment, then the
profile's own settings over the starter.

A starter holds answer-time settings and mode sections only. The deployment's endpoints, keys and roles stay its own,
so a starter that sets one is refused, as is a name the engine does not ship.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mnemiq
from mnemiq.agent.modes import modes_for
from mnemiq.config import ModeSettings, Settings
from mnemiq.profiles import (ProfileFileError, _read, _starter_entry, flattened_entry, load_profiles, profile_sources,
                             profile_starter, profile_version, starter_names, starter_path, starters_folder)
from tests.starter_fixtures import (ROOT_AND_CHILD, STARTER, STARTER_HEAD, STARTER_INSTANT, shipped_starters_untouched,
                                    starter_folder, write)

LOCAL = "http://localhost:8000/v1"

ONLY = 'default = "only"\n[profiles.only]\nstarter = "fixture-2.5-build"\n'

# the default `only` starts from the starter and turns guided decoding on
GUIDED_ROOT = 'default = "only"\n[profiles.only]\nstarter = "fixture-2.5-build"\nguided_sql = true\n'

# the default `root` starts from the starter and names the chat endpoint and its key itself
ROOT_WITH_ENDPOINT = (f'default = "root"\n[profiles.root]\nstarter = "fixture-2.5-build"\nllm_base_url = "{LOCAL}"\n'
                      'api_key_env = "R_KEY"\n')


def _settings(**fields) -> Settings:
    fields.setdefault("llm_base_url", LOCAL)
    fields.setdefault("llm_api_key", "local-key")
    fields.setdefault("llm_model", "base-model")
    return Settings(acme_data_dir=None, **fields)


def _load(tmp_path, text: str, **fields):
    return load_profiles(_settings(profiles_path=str(write(tmp_path / "profiles.toml", text)), **fields))


def test_the_starter_is_over_the_deployment_and_under_the_root(tmp_path, monkeypatch):      # Review Focus 1
    # root: starter "fixture-2.5-build", writer_temperature 0.2; child: extends root, retrieval_k 12
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    profiles, _ = _load(tmp_path, ROOT_AND_CHILD, retrieval_k=5, writer_temperature=0.7)
    root, child = profiles["root"].settings, profiles["child"].settings
    assert (root.retrieval_k, root.writer_temperature) == (36, 0.2)
    assert (child.retrieval_k, child.writer_temperature) == (12, 0.2)
    assert modes_for(child)["instant"].card_style == "bare"


def test_a_starter_cannot_clear_what_it_held_unset(tmp_path, monkeypatch):                    # Review Focus 1
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    assert _load(tmp_path, ONLY, llm_seed=7)[0]["only"].settings.llm_seed == 7


def test_env_mode_settings_never_meet_a_starter(tmp_path, monkeypatch):                       # gate item 1
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    with pytest.raises(ProfileFileError, match="MNEMIQ_PROFILES_PATH and MNEMIQ_MODE_SETTINGS are both set"):
        _load(tmp_path, ONLY, mode_settings={"instant": ModeSettings(prose=False)})


def test_a_starter_resolves_under_a_deployment_with_no_chat_endpoint(tmp_path, monkeypatch):
    # only the root names the chat endpoint: no layer before it may be refused for lacking one
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    monkeypatch.setenv("R_KEY", "r")
    profiles, _ = _load(tmp_path, ROOT_WITH_ENDPOINT, llm_base_url=None, llm_api_key=None,
                        embed_base_url=LOCAL, embed_api_key="e")
    assert profiles["root"].settings.retrieval_k == 36


@pytest.mark.parametrize("name", ["no-such-build", "../profiles", "Qwen2.5"])
def test_an_unknown_starter_is_refused_listing_the_shipped_ones(tmp_path, monkeypatch, name):   # Review Focus 2
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    with pytest.raises(ProfileFileError, match=r"is not one this engine ships; it ships: fixture-2.5-build"):
        _load(tmp_path, f'default = "only"\n[profiles.only]\nstarter = "{name}"\n')


def test_a_link_in_the_starters_folder_is_not_a_starter(tmp_path, monkeypatch):                     # Review Focus 2
    # a valid starter outside the folder, linked into it under its own name: its name must never reach that file
    folder = starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    (folder / "linked.toml").symlink_to(write(tmp_path / "outside.toml", STARTER.replace("fixture-2.5-build", "linked")))
    assert starter_names() == ["fixture-2.5-build"]
    with pytest.raises(ProfileFileError, match=r"starter 'linked' is not one this engine ships; it ships: fixture-2\.5-build$"):
        _load(tmp_path, 'default = "only"\n[profiles.only]\nstarter = "linked"\n')


@pytest.mark.parametrize("line", ['llm_base_url = "http://192.0.2.1/v1"', 'llm_model = "m"', "llm_context_window = 8192",
                                  'verify_model = "j"', 'verify_base_url = "http://192.0.2.1/v1"', 'api_key_env = "K"',
                                  'verify_api_key_env = "K"', 'roles = ["*"]', 'extends = "x"', 'starter = "y"'])
def test_a_starter_that_sets_what_the_deployment_owns_is_refused(tmp_path, monkeypatch, line):
    # the line goes before the .instant header, so it lands in the profile's own table (I4a)
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER_HEAD + line + "\n" + STARTER_INSTANT})
    with pytest.raises(ProfileFileError, match=r"starter 'fixture-2.5-build' sets .*, which a starter never sets"):
        _load(tmp_path, ONLY)


def test_a_value_the_model_cannot_take_is_refused_naming_where_it_was_set(tmp_path, monkeypatch):   # Review Focus 3
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    with pytest.raises(ProfileFileError, match=r"'only' sets writer_temperature on 'gpt-5', which takes none "
                                               r"\(the value comes from starter 'fixture-2.5-build'\)"):
        _load(tmp_path, ONLY, llm_model="gpt-5")


def test_the_reasoning_refusal_names_where_each_of_its_two_settings_came_from(tmp_path, monkeypatch):   # M9
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER_HEAD + 'writer_reasoning = "before_sql"\n' + STARTER_INSTANT})
    with pytest.raises(ProfileFileError, match=r"\(writer_reasoning from starter 'fixture-2.5-build', guided_sql from profile 'only'\)"):
        _load(tmp_path, GUIDED_ROOT)


def test_with_no_profiles_file_the_refusals_read_as_before():                                         # M9
    with pytest.raises(ProfileFileError, match=r"'default' sets writer_temperature on 'gpt-5', which takes none$"):
        load_profiles(_settings(llm_model="gpt-5", writer_temperature=0.0))


def test_a_starters_change_gives_its_profiles_a_new_version(tmp_path, monkeypatch):
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    before = profile_version(_load(tmp_path, ONLY)[0]["only"].settings, "duckdb")
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER.replace("retrieval_k = 36", "retrieval_k = 40")})
    assert profile_version(_load(tmp_path, ONLY)[0]["only"].settings, "duckdb") != before


def test_flattened_entry_and_sources_follow_the_chain(tmp_path, monkeypatch):
    folder = starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    path = str(write(tmp_path / "chain.toml", ROOT_AND_CHILD))
    entry = flattened_entry(path, "child")
    assert (entry.retrieval_k, entry.writer_temperature, entry.instant.card_style) == (12, 0.2, "bare")
    assert entry.model_fields_set == {"retrieval_k", "writer_temperature", "instant"}         # the layers' union (M8)
    assert profile_sources(path, "child") == [path, str(folder / "fixture-2.5-build.toml")]
    assert profile_starter(path, "child") == "fixture-2.5-build"


@pytest.mark.parametrize("evidence", [None, "not json", "[]", "{}", '{"starter_sha256": "' + "0" * 64 + '"}'])
def test_a_starter_whose_evidence_is_missing_or_names_another_file_is_refused(tmp_path, monkeypatch, evidence):
    # what an interrupted publish leaves: a starter beside no evidence, or beside the evidence of another file
    folder = starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    path = folder / "fixture-2.5-build.evidence.json"
    path.unlink() if evidence is None else path.write_text(evidence)
    with pytest.raises(ProfileFileError, match=r"starter 'fixture-2\.5-build' and its evidence do not match .*: it may "
                                               r"have been published incompletely\. Restore both files, or publish it "
                                               r"again"):
        _load(tmp_path, ONLY)


def test_a_starter_bound_to_its_evidence_loads(tmp_path, monkeypatch):
    # positive control: the evidence names the file's sha256, which is all the loader reads of it
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    assert _load(tmp_path, ONLY)[0]["only"].settings.retrieval_k == 36


def test_every_shipped_starter_is_one_profile_named_for_its_file():
    # vacuous by design until PR C ships the first starter; Task 6 adds `assert starter_names()` here (gate item 10)
    for name in starter_names():                 # a loop, not a parametrization, which would skip without a word
        read = _read(starter_path(name))
        assert read.default == name and list(read.profiles) == [name]
        _starter_entry(name)


def test_the_starters_folder_is_inside_the_installed_package():                     # Review Focus 5
    import mnemiq
    assert starters_folder() == Path(mnemiq.__file__).resolve().parent / "starter_profiles"
    assert (starters_folder() / "__init__.py").is_file()


def test_the_guard_on_the_shipped_folder_catches_a_write():
    # positive control for I5's autouse fixture, through the context manager it wraps
    probe = Path(mnemiq.__file__).resolve().parent / "starter_profiles" / "probe.toml"
    with pytest.raises(AssertionError, match="probe.toml"):
        with shipped_starters_untouched():
            probe.write_text("default = 'x'\n")
    assert not probe.exists()
