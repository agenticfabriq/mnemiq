"""Two writer settings: room to reason before the SQL, and the writer's prompt without the engine's rule text.

Both are techniques other text-to-SQL systems use, written here in the engine's own words, so a sweep can say which one
a model and dataset gain from. Each leaves the default's bytes alone, and every place that renders the writer's system
prompt (the writer, the prompt fitter, the window check and the profile's version) renders it from the same settings.
"""

from __future__ import annotations

import duckdb
import pytest

from mnemiq.agent.loop import STRATEGIES
from mnemiq.config import Settings
from mnemiq.contract import Column, Snapshot, SourceBinding
from mnemiq.generate.generator import LLMGenerator
from mnemiq.generate.prompts import PERSONA, system_prompt, writer_style
from mnemiq.profiles import PROFILE_SETTINGS, ProfileFileError, _prompt_texts, load_profiles, profile_version
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard

_ANSWER = '{"sql": "SELECT n FROM claim", "reason": "r"}'


class _Client:
    def __init__(self, reply: str = _ANSWER):
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, system, user, max_tokens=512, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        return self.reply


def _packet() -> ContextPacket:
    return ContextPacket(question="how many claims?", cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
                         grant_fingerprint="g", enrichment_version="v1")


def _settings(**fields) -> Settings:
    fields.setdefault("llm_base_url", "http://localhost:8000/v1")
    fields.setdefault("llm_api_key", "k")
    fields.setdefault("llm_model", "qwen")
    return Settings(acme_data_dir=None, **fields)


def _propose(reply: str, **generator_fields):
    client = _Client(reply)
    return LLMGenerator(client, **generator_fields).propose(_packet()), client


# --- the defaults ------------------------------------------------------------------------------------------------------


def test_the_defaults_render_the_prompt_as_before():
    assert system_prompt(reasoning="none", rules="full") == system_prompt()
    for strategy in STRATEGIES:
        assert system_prompt(strategy=strategy, reasoning="none", rules="full") == system_prompt(strategy=strategy)


def test_the_defaults_are_none_and_full():
    settings = _settings()
    assert settings.writer_reasoning == "none" and settings.writer_rules == "full"


@pytest.mark.parametrize("field, value", [("writer_reasoning", "always"), ("writer_rules", "some")])
def test_an_unknown_value_is_refused(field, value):
    with pytest.raises(ValueError):
        _settings(**{field: value})


def test_both_are_a_profiles_to_change():
    assert {"writer_reasoning", "writer_rules"} <= PROFILE_SETTINGS


# --- reasoning before the SQL ------------------------------------------------------------------------------------------


def test_reasoning_asks_for_the_thinking_first_and_the_json_last():
    text = system_prompt(reasoning="before_sql")
    assert "step by step" in text
    assert "no prose" not in text
    assert '{"sql": "<the SELECT, or null>"' in text


def test_the_json_after_the_reasoning_is_read():
    reply = 'The question counts claims {all of them}, so one table.\n\n' + _ANSWER
    proposal, _ = _propose(reply, reasoning="before_sql")
    assert proposal.sql == "SELECT n FROM claim"


def test_the_object_that_ends_the_reply_is_the_answer():
    reply = ('A reply looks like {"sql": "SELECT 1", "reason": "draft"}, but the table is claim.\n'
             '```json\n' + _ANSWER + '\n```')
    proposal, _ = _propose(reply, reasoning="before_sql")
    assert proposal.sql == "SELECT n FROM claim"


def test_braces_inside_the_sql_do_not_end_the_object():
    reply = 'Thinking.\n{"sql": "SELECT \'{x}\' AS s FROM claim", "reason": "r"}'
    proposal, _ = _propose(reply, reasoning="before_sql")
    assert proposal.sql == "SELECT '{x}' AS s FROM claim"


def test_an_object_nested_in_the_answer_is_not_the_answer():
    reply = 'Thinking.\n{"sql": "SELECT n FROM claim", "reason": "r", "notes": {"grain": "one row"}}'
    proposal, _ = _propose(reply, reasoning="before_sql")
    assert proposal.sql == "SELECT n FROM claim"


def test_a_draft_before_a_cut_off_answer_is_never_run():
    """A reply cut short by the token limit ends mid-answer. The draft before it was abandoned, not answered."""
    reply = 'Draft: {"sql": "SELECT 1", "reason": "draft"}. Better:\n{"sql": "SELECT n FROM cl'
    proposal, _ = _propose(reply, reasoning="before_sql")
    assert proposal.sql is None


def test_a_cut_off_answer_that_ends_on_a_brace_does_not_bring_back_the_draft():
    reply = ('Draft: {"sql": "SELECT 1", "reason": "draft"}. Better:\n'
             '{"sql": "SELECT n FROM claim WHERE note = \'}')
    proposal, _ = _propose(reply, reasoning="before_sql")
    assert proposal.sql is None


@pytest.mark.parametrize("reply", [
    '{"sql": null, "reason": "no table", "assumed_terms": ["profit"], "notes": {"sql": "SELECT 1"}',
    'Thinking.\n{\n  "sql": null,\n  "reason": "no table",\n  "notes": {\n    "sql": "SELECT 1"\n  }',
])
def test_a_cut_off_decline_does_not_become_the_query_nested_in_it(reply):
    """The outer object lost its closing brace to the token limit, so the reply ends on the nested one's."""
    proposal, _ = _propose(reply, reasoning="before_sql", declare_assumed_terms=True)
    assert proposal.sql is None


def test_a_broken_answer_does_not_become_the_object_inside_it():
    """An unquoted quote breaks the outer object partway; the object after the break is inside it, not an answer."""
    proposal, _ = _propose('Thinking.\n{"sql": null, "reason": "the "x" {"sql": "SELECT 1"}', reasoning="before_sql")
    assert proposal.sql is None


def test_a_brace_in_the_reasoning_does_not_hide_the_answer_after_it():
    reply = 'The grouping is like {"k": [1, 2\nso one table.\n' + _ANSWER
    proposal, _ = _propose(reply, reasoning="before_sql")
    assert proposal.sql == "SELECT n FROM claim"


def test_a_decline_with_sql_nested_inside_stays_a_decline():
    reply = 'Thinking.\n{"sql": null, "reason": "no table", "assumed_terms": ["profit"], "notes": {"sql": "SELECT 1"}}'
    proposal, _ = _propose(reply, reasoning="before_sql", declare_assumed_terms=True)
    assert proposal.sql is None
    assert proposal.reason == "no table" and proposal.assumed_terms == ["profit"]


def test_a_decline_after_reasoning_keeps_its_reason():
    proposal, _ = _propose('No table holds prices.\n{"sql": null, "reason": "no price table"}', reasoning="before_sql")
    assert proposal.sql is None and proposal.reason == "no price table"


def test_assumed_terms_survive_the_reasoning():
    reply = 'Profit is not defined.\n{"sql": "SELECT n FROM claim", "reason": "r", "assumed_terms": ["profit"]}'
    proposal, _ = _propose(reply, reasoning="before_sql", declare_assumed_terms=True)
    assert proposal.sql == "SELECT n FROM claim" and proposal.assumed_terms == ["profit"]


def test_words_after_the_answer_make_it_unread():
    """The prompt asks for nothing after the JSON. Text after it could be a second thought, so it is not guessed at."""
    proposal, _ = _propose(_ANSWER + "\nThis counts every claim.", reasoning="before_sql")
    assert proposal.sql is None and proposal.reason


def test_reasoning_with_no_json_is_a_deferral_not_an_error():
    proposal, _ = _propose("I thought about it and ran out of room", reasoning="before_sql")
    assert proposal.sql is None and proposal.reason


def test_the_default_reader_is_unchanged():
    # Before the setting, braces in prose ahead of the JSON made the reply unreadable, and a deferral. Off, it still is.
    proposal, _ = _propose('Thinking {aloud}.\n' + _ANSWER)
    assert proposal.sql is None


def test_reasoning_drops_silently_from_the_strategies():
    for strategy in ("decompose", "skeleton"):
        assert "silently" in system_prompt(strategy=strategy)
        assert "silently" not in system_prompt(strategy=strategy, reasoning="before_sql")


def test_the_generator_sends_the_reasoning_prompt():
    _, client = _propose(_ANSWER, reasoning="before_sql")
    assert client.calls[0]["system"] == system_prompt(reasoning="before_sql")


@pytest.mark.parametrize("with_file", [False, True])
def test_reasoning_under_guided_decoding_is_refused(tmp_path, with_file):
    """Guided decoding holds the reply to the JSON from its first token, so the reasoning could never be written."""
    if with_file:
        path = tmp_path / "p.toml"
        path.write_text('default = "only"\n[profiles.only]\nguided_sql = true\nwriter_reasoning = "before_sql"\n')
        settings = _settings(profiles_path=str(path))
    else:
        settings = _settings(guided_sql=True, writer_reasoning="before_sql")
    with pytest.raises(ProfileFileError, match="writer_reasoning.*guided_sql|guided_sql.*writer_reasoning"):
        load_profiles(settings)


# --- the prompt without the engine's rule text -------------------------------------------------------------------------


def test_no_rules_drops_the_persona_and_the_rule_sections():
    text = system_prompt(rules="none")
    assert PERSONA not in text
    assert "HARD RULES" not in text and "CODED VALUES" not in text


def test_no_rules_keeps_the_task_the_decline_and_the_reply():
    text = system_prompt(rules="none")
    assert "Write one duckdb SELECT query that answers the question." in text
    assert "DEFERRING IS A CORRECT ANSWER" in text
    assert '{"sql": "<the SELECT, or null>"' in text


def test_no_rules_keeps_the_assertive_decline_when_set():
    text = system_prompt(rules="none", assertive=True)
    assert "ATTEMPT EVERY QUESTION" in text and "DEFERRING IS A CORRECT ANSWER" not in text


def test_both_together():
    text = system_prompt(rules="none", reasoning="before_sql")
    assert "HARD RULES" not in text and "step by step" in text


# --- one rendering everywhere ------------------------------------------------------------------------------------------

_CHANGED = {"writer_reasoning": "before_sql", "writer_rules": "none", "assertive_sql": True}


def _expected(strategy=None) -> str:
    return system_prompt(dialect="duckdb", strategy=strategy, **writer_style(_settings(**_CHANGED)))


def test_the_style_reads_every_setting_that_changes_the_writers_system_prompt():
    assert writer_style(_settings(**_CHANGED, guard_undefined_terms=True)) == {
        "assertive": True, "declare_assumed_terms": True, "reasoning": "before_sql", "rules": "none"}


def test_the_assembled_writer_renders_it():
    from mnemiq.assembly import build_components

    class _Adapter:
        dialect = "duckdb"

    kit = build_components(_settings(**_CHANGED), _Adapter(), duckdb.connect())
    client = _Client()
    kit.generator._client = client
    kit.generator.propose(_packet())
    assert client.calls[0]["system"] == _expected()


def test_the_fitter_renders_it():
    from mnemiq.semantic.fit import fitter_for

    assert fitter_for(_settings(**_CHANGED), "duckdb").system() == max(
        (_expected(s) for s in (None, *STRATEGIES)), key=len)


def test_the_window_check_renders_it():
    from mnemiq.llm import window

    snap = Snapshot(version="v", source_id="s", created_at="t",
                    columns=[Column(id="claim.n", object_id="claim", name="n", data_type="INTEGER")],
                    source_bindings=[SourceBinding(id="claim", source_id="s", object_id="claim", source_object="claim",
                                                   binding_type="table")])
    system, *_ = window.largest_prompts(duckdb.connect(), snap, _settings(**_CHANGED), "duckdb", window.everything(snap))
    assert system == max((_expected(s) for s in (None, *STRATEGIES)), key=len)


def test_every_writer_construction_site_passes_both_settings():
    """A site that left them out would run the default prompt and reader under a profile that set them."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sites = []
    for path in [*root.joinpath("src").rglob("*.py"), *root.joinpath("scripts").rglob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "LLMGenerator":
                named = {k.arg for k in node.keywords}
                sites.append((str(path.relative_to(root)), {"reasoning", "rules"} - named))
    assert sites, "the scan found no site, so it checks nothing"
    assert [site for site in sites if site[1]] == []


def test_the_version_hashes_it():
    texts = _prompt_texts(_settings(**_CHANGED), "duckdb")
    for strategy in STRATEGIES:
        assert texts[f"writer system {strategy}"] == _expected(strategy)


@pytest.mark.parametrize("field, value", [("writer_reasoning", "before_sql"), ("writer_rules", "none")])
def test_each_changes_the_version(field, value):
    assert profile_version(_settings(), "duckdb") != profile_version(_settings(**{field: value}), "duckdb")
