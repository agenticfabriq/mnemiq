"""Writer temperature, the prompt's layout, the decider's repairs and the prose call are settings.

Each changes what it names, and leaves the default's bytes alone. Measured runs asked for them: the prose call is most of
instant's speed gap on BIRD, the layout moves which tables a writer reads, and the writer's temperature has never been tested
on its own.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.agent.modes import MODES, build_agent, modes_for
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import ModeSettings, Settings
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.generate.generator import FakeGenerator, LLMGenerator
from mnemiq.generate.prompts import question_last_prompt, system_prompt, user_prompt
from mnemiq.profiles import PROFILE_SETTINGS, ProfileFileError, load_profiles, profile_version
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_STAR = '{"sql": "SELECT * FROM claim", "reason": "r"}'
_GOOD = '{"sql": "SELECT n FROM claim", "reason": "r"}'


class _Client:
    def __init__(self):
        self.calls: list[dict] = []

    def complete(self, system, user, max_tokens=512, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        return '{"sql": "SELECT n FROM claim", "reason": "r"}'


class _Adapter:
    dialect = "duckdb"

    def __init__(self, table=None):
        self.table = table if table is not None else pa.table({"n": [7]})

    def execute_arrow(self, sql, timeout_s=None):
        return self.table

    def execute(self, sql):
        return []


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="acme", created_at="t", columns=[Column(id="claim.n", object_id="claim", name="n")])


def _packet() -> ContextPacket:
    return ContextPacket(question="how many claims?", cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
                         grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")


def _settings(**fields) -> Settings:
    fields.setdefault("llm_base_url", "http://localhost:8000/v1")
    fields.setdefault("llm_api_key", "k")
    fields.setdefault("llm_model", "qwen")
    return Settings(acme_data_dir=None, **fields)


def _agent(replies, synth=None, table=None, **mode_fields):
    mode = modes_for(_settings(mode_settings={"instant": mode_fields}))["instant"] if mode_fields else MODES["instant"]
    return build_agent(mode, generator=FakeGenerator(replies), synthesizer=synth or FakeSynthesizer("prose"),
                       adapter=_Adapter(table), cache=TwoTierCache(L1Cache()), corrector=None, values=None, selector=None)


def _ask(agent):
    return agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)


def test_the_two_new_settings_are_a_profiles_to_change():
    assert {"writer_temperature", "prompt_layout"} <= PROFILE_SETTINGS


# --- writer temperature ----------------------------------------------------------------------------------------------


def test_a_writer_temperature_reaches_the_writer():
    client = _Client()
    LLMGenerator(client, temperature=0.0).propose(_packet())
    assert client.calls[0]["temperature"] == 0.0


def test_unset_no_temperature_is_sent():
    client = _Client()
    LLMGenerator(client).propose(_packet())
    assert "temperature" not in client.calls[0]


def test_it_reaches_only_the_writer():
    """The corrector and the prose are other roles. Built through the real assembly, so the join is what is tested."""
    from mnemiq.assembly import build_components
    from mnemiq.llm.client import LLMClient

    sent: list[float | None] = []

    def complete(self, system, user, max_tokens=512, extra_body=None, temperature=None):
        sent.append(temperature)
        return '{"sql": "SELECT n FROM claim", "reason": "r"}'

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LLMClient, "complete", complete)
        kit = build_components(_settings(writer_temperature=0.0), _Adapter(), duckdb.connect())
        kit.generator.propose(_packet())
        kit.corrector.correct("SELECT 1", "p")
        kit.synthesizer.answer("q", "s", "r")
    assert sent == [0.0, None, None]


@pytest.mark.parametrize("model", ["gpt-5.5", "openai.gpt-5.5", "o3"])
def test_a_temperature_the_model_would_ignore_is_refused(model):
    """`LLMClient` drops it for these, so it would boot, get a version and log a value the model never sampled at."""
    with pytest.raises(ProfileFileError, match="writer_temperature"):
        load_profiles(_settings(llm_model=model, writer_temperature=0.0))


def test_a_temperature_the_model_takes_is_kept():
    profiles, _ = load_profiles(_settings(writer_temperature=0.0))
    assert profiles["default"].settings.writer_temperature == 0.0


def test_the_refusal_applies_to_a_profile_in_a_file(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('default = "only"\n[profiles.only]\nllm_model = "gpt-5.5"\nwriter_temperature = 0.0\n')
    with pytest.raises(ProfileFileError, match="only.*writer_temperature"):
        load_profiles(_settings(profiles_path=str(path)))


# --- the prompt's layout ---------------------------------------------------------------------------------------------


def test_question_last_puts_the_context_in_the_system_message():
    system, user = question_last_prompt(system_prompt(), _packet())
    assert "TABLE claim" in system and "TABLE claim" not in user
    assert user == "QUESTION: how many claims?"


def test_question_last_keeps_the_feedback_after_the_question():
    _, user = question_last_prompt(system_prompt(), _packet(), "fix this")
    assert user.startswith("QUESTION: how many claims?")
    assert user.endswith("fix this")


def test_the_generator_sends_the_layout_it_was_given():
    client = _Client()
    LLMGenerator(client, layout="question_last").propose(_packet())
    assert client.calls[0]["user"] == "QUESTION: how many claims?"
    assert "TABLE claim" in client.calls[0]["system"]


def test_the_default_layout_sends_what_it_always_sent():
    client = _Client()
    LLMGenerator(client).propose(_packet())
    assert client.calls[0]["user"] == user_prompt(_packet())
    assert client.calls[0]["system"] == system_prompt()


def test_the_layout_reaches_the_generator_through_the_assembly():
    from mnemiq.assembly import build_components

    kit = build_components(_settings(prompt_layout="question_last"), _Adapter(), duckdb.connect())
    assert kit.generator._layout == "question_last"


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError):
        _settings(prompt_layout="sideways")


def test_the_window_sees_the_same_text_either_way():
    """The window check and the fit count the default's text. The layout moves it and adds only a separator, which their
    margin covers, so neither needs to know the layout."""
    system = system_prompt()
    first = len(system) + len(user_prompt(_packet()))
    moved, user = question_last_prompt(system, _packet())
    assert abs((len(moved) + len(user)) - first) <= 8


def test_the_layout_changes_the_version_through_the_prompt_as_well_as_the_setting(monkeypatch):
    from mnemiq.profiles import _prompt_texts

    first = _prompt_texts(_settings(), "duckdb")
    last = _prompt_texts(_settings(prompt_layout="question_last"), "duckdb")
    assert first["writer user"] != last["writer user"]
    assert profile_version(_settings(), "duckdb") != profile_version(_settings(prompt_layout="question_last"), "duckdb")


# --- the decider's repairs -------------------------------------------------------------------------------------------


def test_the_default_repairs_a_refusal_up_to_three_writer_calls():
    agent = _agent([_STAR, _STAR, _GOOD])
    assert not _ask(agent).deferred
    assert len(agent.generator.calls) == 3


def test_plan_attempts_caps_the_writer_calls():
    agent = _agent([_STAR, _STAR, _GOOD], plan_attempts=1)
    answer = _ask(agent)
    assert answer.deferred
    assert len(agent.generator.calls) == 1


def test_plan_attempts_of_two():
    agent = _agent([_STAR, _GOOD], plan_attempts=2)
    assert not _ask(agent).deferred
    assert len(agent.generator.calls) == 2


def test_deep_candidates_stay_at_one_call_each():
    """A refused candidate drops out of the vote, however many repairs a mode's single path may spend."""
    agent = Agent(generator=FakeGenerator([_STAR] + [_GOOD] * 9), synthesizer=FakeSynthesizer("p"), adapter=_Adapter(),
                  cache=TwoTierCache(L1Cache()), budget=Budget(), candidates=5, plan_attempts=3)
    _ask(agent)
    assert len(agent.generator.calls) == 5


def test_plan_attempts_must_be_positive():
    with pytest.raises(ValueError):
        ModeSettings(plan_attempts=0)


# --- the prose call --------------------------------------------------------------------------------------------------


def test_prose_off_makes_no_model_call_and_states_the_row_count():
    synth = FakeSynthesizer("prose")
    answer = _ask(_agent([_GOOD], synth=synth, prose=False))
    assert synth.calls == []
    assert answer.answer.startswith("Returned 1 row.")
    assert answer.preview is not None and answer.preview.rows == [[7]]


def test_prose_off_counts_rows_in_the_plural():
    answer = _ask(_agent([_GOOD], table=pa.table({"n": [1, 2, 3]}), prose=False))
    assert answer.answer.startswith("Returned 3 rows.")


def test_prose_off_keeps_the_engines_own_notes():
    """The cut note and the "none" note are the engine's words, not the model's, and go on every answer."""
    from mnemiq.agent import loop

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loop, "cap_sentence", lambda cap: "CUT-NOTE")
        patch.setattr(loop, "_cut", lambda approved, table: 1)
        answer = _ask(_agent([_GOOD], prose=False))
    assert "CUT-NOTE" in answer.answer


def test_prose_off_on_a_cached_answer():
    synth = FakeSynthesizer("prose")
    agent = _agent([_GOOD, _GOOD], synth=synth, prose=False)
    _ask(agent)
    second = _ask(agent)
    assert second.cached is True
    assert second.answer.startswith("Returned 1 row.")
    assert synth.calls == []


def test_prose_off_keeps_a_decline_as_it_was():
    plain = _ask(_agent(['{"sql": null, "reason": "no table"}']))
    off = _ask(_agent(['{"sql": null, "reason": "no table"}'], prose=False))
    assert off.deferred and plain.deferred
    assert off.answer == plain.answer


def test_prose_on_is_the_default_and_calls_the_model():
    synth = FakeSynthesizer("prose")
    assert _ask(_agent([_GOOD], synth=synth)).answer.startswith("prose")
    assert len(synth.calls) == 1


def test_each_setting_changes_the_version():
    base = profile_version(_settings(), "duckdb")
    for fields in ({"writer_temperature": 0.0}, {"mode_settings": {"instant": {"plan_attempts": 1}}},
                   {"mode_settings": {"instant": {"prose": False}}}):
        assert profile_version(_settings(**fields), "duckdb") != base
