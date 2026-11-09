"""A profile can extend another in its file: the parent's settings first, then the child's own over them.

Each layer is resolved over the one before it, by the rules a profile follows over the deployment, so a parent is to
its child what the deployment is to a profile with no parent. Roles, the description and being the default are never
inherited.
"""

from __future__ import annotations

import pytest

from mnemiq.agent.modes import modes_for
from mnemiq.config import Settings
from mnemiq.profiles import ProfileFileError, load_profiles, profile_version
from tests.starter_fixtures import STARTER, starter_folder, write

LOCAL = "http://localhost:8000/v1"
E = "https://e.example.com/v1"
F = "https://f.example.net/v1"

# parent: retrieval_k 36, writer_temperature 0.0, instant card_style bare and prose false; child: extends parent,
# retrieval_k 12, instant prose true
PARENT_CHILD = (
    'default = "parent"\n'
    '[profiles.parent]\nretrieval_k = 36\nwriter_temperature = 0.0\n'
    '[profiles.parent.instant]\ncard_style = "bare"\nprose = false\n'
    '[profiles.child]\nroles = ["*"]\nextends = "parent"\nretrieval_k = 12\n'
    '[profiles.child.instant]\nprose = true\n'
)

# the child's resolved values, held by one profile with no parent
FLAT_COPY_OF_CHILD = (
    '[profiles.flat]\nroles = ["*"]\nretrieval_k = 12\nwriter_temperature = 0.0\n'
    '[profiles.flat.instant]\ncard_style = "bare"\nprose = true\n'
)

# c (root): retrieval_k 16, writer_temperature 0.0; b extends c: retrieval_k 20; a extends b. Written leaf first, so
# the file's order is not the chain's.
CHAIN = (
    'default = "c"\n'
    '[profiles.a]\nroles = ["*"]\nextends = "b"\n'
    '[profiles.b]\nroles = ["*"]\nextends = "c"\nretrieval_k = 20\n'
    '[profiles.c]\nretrieval_k = 16\nwriter_temperature = 0.0\n'
)

CYCLE = 'default = "a"\n[profiles.a]\nextends = "b"\n[profiles.b]\nroles = ["*"]\nextends = "a"\n'
SELF = 'default = "a"\n[profiles.a]\nextends = "a"\n'
UNKNOWN_PARENT = 'default = "child"\n[profiles.child]\nextends = "nope"\n'
TWO_PARENTS = (
    'default = "parent"\n[profiles.parent]\nretrieval_k = 36\n'
    '[profiles.child]\nroles = ["*"]\nextends = "parent"\nstarter = "fixture-2.5-build"\n'
)

# the default `other`, a `parent` that lists its roles, and `child`, extending `parent`, listing none
CHILD_WITHOUT_ROLES = (
    'default = "other"\n[profiles.other]\nretrieval_k = 8\n'
    '[profiles.parent]\nroles = ["analyst"]\nretrieval_k = 36\n'
    '[profiles.child]\nextends = "parent"\n'
)

# the default `parent`, with no roles as a default takes, and `child` extending it with none
CHILD_OF_DEFAULT = 'default = "parent"\n[profiles.parent]\nretrieval_k = 36\n[profiles.child]\nextends = "parent"\n'

# parent: endpoint E, E_KEY, llm_context_window 8192, verify_model "judge-e"; child: extends parent, endpoint F, F_KEY
MOVES = (
    'default = "parent"\n'
    f'[profiles.parent]\nllm_base_url = "{E}"\napi_key_env = "E_KEY"\nllm_context_window = 8192\n'
    'verify_model = "judge-e"\n'
    f'[profiles.child]\nroles = ["*"]\nextends = "parent"\nllm_base_url = "{F}"\napi_key_env = "F_KEY"\n'
)

# MOVES with the child's endpoint and key removed: the child sets nothing of its own
STAYS = MOVES.replace(f'llm_base_url = "{F}"\napi_key_env = "F_KEY"\n', "")

# parent: endpoint E, E_KEY; child: extends parent and sets the chat endpoint back to the deployment's own, naming no key
BACK_HOME = (
    'default = "parent"\n'
    f'[profiles.parent]\nllm_base_url = "{E}"\napi_key_env = "E_KEY"\n'
    f'[profiles.child]\nroles = ["*"]\nextends = "parent"\nllm_base_url = "{LOCAL}"\n'
)

# the default `a`, with a mode section, and `b`; neither has a parent
NO_PARENT_FILE = (
    'default = "a"\n'
    '[profiles.a]\nretrieval_k = 36\n'
    '[profiles.a.instant]\ncard_style = "bare"\nprose = false\n'
    '[profiles.b]\nroles = ["*"]\nwriter_temperature = 0.0\n'
)

# Recorded on main at 67f6ee73, before profiles could have parents, by running this test's dictionary over
# NO_PARENT_FILE. A profile with no parent must keep the version it had.
GOLDEN_NO_PARENT_VERSIONS = {"a": "65a67540709a", "b": "91c4817da541"}


def _settings(**fields) -> Settings:
    fields.setdefault("llm_base_url", LOCAL)
    fields.setdefault("llm_api_key", "local-key")
    fields.setdefault("llm_model", "base-model")
    return Settings(acme_data_dir=None, **fields)


def _load(tmp_path, text: str, **fields):
    return load_profiles(_settings(profiles_path=str(write(tmp_path / "profiles.toml", text)), **fields))


def test_a_child_takes_its_parents_settings_and_overrides_its_own(tmp_path):
    # parent: retrieval_k 36, writer_temperature 0.0; child: extends parent, retrieval_k 12
    child = _load(tmp_path, PARENT_CHILD)[0]["child"].settings
    assert (child.retrieval_k, child.writer_temperature) == (12, 0.0)


def test_a_mode_section_merges_setting_by_setting(tmp_path):
    # parent.instant: card_style bare, prose false; child.instant: prose true
    instant = modes_for(_load(tmp_path, PARENT_CHILD)[0]["child"].settings)["instant"]
    assert (instant.card_style, instant.prose) == ("bare", True)


@pytest.mark.parametrize("text", [CHILD_WITHOUT_ROLES, CHILD_OF_DEFAULT])
def test_roles_and_default_are_never_inherited(tmp_path, text):
    # a child of the default is not the default, so it lists its own roles
    with pytest.raises(ProfileFileError, match="'child' must list the roles"):
        _load(tmp_path, text)


def test_a_chain_resolves_from_the_root_down(tmp_path):
    # c (root): retrieval_k 16, writer_temperature 0.0; b extends c: retrieval_k 20; a extends b
    a = _load(tmp_path, CHAIN)[0]["a"].settings
    assert (a.retrieval_k, a.writer_temperature) == (20, 0.0)


@pytest.mark.parametrize("text, match", [
    (CYCLE, r"'a' extends a chain that comes back to itself: a -> b -> a"),
    (SELF, r"'a' extends a chain that comes back to itself: a -> a"),
    (UNKNOWN_PARENT, r"'child' extends 'nope', which this file does not hold"),
    (TWO_PARENTS, r"'child' names two parents, extends 'parent' and starter 'fixture-2.5-build': a profile has at most one"),
])
def test_a_chain_that_cannot_resolve_is_refused_at_boot(tmp_path, monkeypatch, text, match):
    starter_folder(tmp_path, monkeypatch, {"fixture-2.5-build": STARTER})
    with pytest.raises(ProfileFileError, match=match):
        _load(tmp_path, text)


def test_a_child_that_moves_drops_its_parents_window_and_judge(tmp_path, monkeypatch):
    # parent: endpoint E, E_KEY, llm_context_window 8192, verify_model "judge-e"; child: endpoint F, F_KEY
    monkeypatch.setenv("E_KEY", "e")
    monkeypatch.setenv("F_KEY", "f")
    child = _load(tmp_path, MOVES)[0]["child"].settings
    assert (child.llm_base_url, child.llm_api_key) == (F, "f")
    assert child.llm_context_window is None and child.verify_model is None and child.verify_api_key is None


def test_a_child_that_moves_with_no_key_is_refused_naming_it(tmp_path, monkeypatch):
    monkeypatch.setenv("E_KEY", "e")
    with pytest.raises(ProfileFileError, match="'child' has no chat endpoint and key"):
        _load(tmp_path, MOVES.replace('api_key_env = "F_KEY"\n', ""))


def test_a_child_back_at_the_deployments_endpoint_needs_its_own_key(tmp_path, monkeypatch):
    # layering compares the child with its parent, so going back is a move: refused, failing closed
    monkeypatch.setenv("E_KEY", "e")
    with pytest.raises(ProfileFileError, match="'child' has no chat endpoint and key"):
        _load(tmp_path, BACK_HOME)


def test_a_child_that_stays_keeps_its_parents_endpoint_key_and_window(tmp_path, monkeypatch):
    monkeypatch.setenv("E_KEY", "e")
    child = _load(tmp_path, STAYS)[0]["child"].settings
    assert (child.llm_base_url, child.llm_api_key, child.llm_context_window) == (E, "e", 8192)


def test_a_parents_change_gives_its_child_a_new_version(tmp_path):
    before = profile_version(_load(tmp_path, PARENT_CHILD)[0]["child"].settings, "duckdb")
    changed = PARENT_CHILD.replace("writer_temperature = 0.0", "writer_temperature = 0.3")
    assert profile_version(_load(tmp_path, changed)[0]["child"].settings, "duckdb") != before


def test_a_child_and_its_flat_copy_share_a_version(tmp_path):
    profiles = _load(tmp_path, PARENT_CHILD + FLAT_COPY_OF_CHILD)[0]
    assert profile_version(profiles["child"].settings, "duckdb") == profile_version(profiles["flat"].settings, "duckdb")


def test_a_profile_with_no_parent_keeps_todays_version(tmp_path):
    # a fixed file of two profiles, one with mode sections; the versions were recorded on main before this change
    profiles = _load(tmp_path, NO_PARENT_FILE)[0]
    assert {name: profile_version(p.settings, "duckdb") for name, p in profiles.items()} == GOLDEN_NO_PARENT_VERSIONS
