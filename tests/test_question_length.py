"""M147: the question reaches the generator whole, or the agent declines it. It is never cut.

`user_prompt` wrote `QUESTION: {sanitize(packet.question, limit=500)}` while the judge, the candidate selector and the
synthesizer read the question whole, so a longer question was answered as a cut one and then checked against all of
it. On BEAVER's 200 sampled questions 84 reached the generator cut, and the annotations its harness appended to the
question almost never arrived (domain knowledge on 0 of 125 questions) while the judge read every word. Nothing said
so: the only witness was the model, which replied on 11 questions that the question looked truncated.

Now every stage reads the same question. Up to MNEMIQ_MAX_QUESTION_CHARS the generator gets it whole; past it the
agent declines before the generator is called, with a reason that says what to do (shorten it, or raise the limit).
"""

import ast
import pathlib

import pyarrow as pa
import pytest

from mnemiq.agent.budget import Budget
from mnemiq.agent.loop import Agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import DEFAULT_MAX_QUESTION_CHARS, Settings
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.contract.seams import DeferralReason
from mnemiq.generate.generator import FakeGenerator
from mnemiq.generate.prompts import user_prompt
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.sql.verdict import Approved
from mnemiq.verify.verifier import Verifier

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_REPLY = '{"sql": "SELECT count(*) AS n FROM claim", "reason": "ok"}'


def _question(chars: int) -> str:
    """Exactly `chars` characters ending in a marker, so a cut anywhere is a missing tail. Single spaces only:
    `sanitize` folds runs of whitespace, and a folded space would read as a mismatch that is not a cut."""
    tail = " ENDOFQUESTION"
    body = ("abcdefghi " * (chars // 10 + 1))[: chars - len(tail)]
    if body.endswith(" "):
        body = body[:-1] + "z"
    text = body + tail
    assert len(text) == chars
    return text


def _packet(question: str) -> ContextPacket:
    return ContextPacket(
        question=question,
        cards=[RetrievedCard(object_id="claim", card="TABLE claim\n- n integer", score=1.0)],
        grant_fingerprint=_GRANTS.fingerprint,
        enrichment_version="v1",
    )


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="acme", created_at="2026-10-12T00:00:00Z",
                    columns=[Column(id="claim.n", object_id="claim", name="n")])


class _Adapter:
    def execute_arrow(self, sql, timeout_s=None):
        return pa.table({"n": [7]})

    def execute(self, sql):  # EXPLAIN, inside decide()
        return []


def _agent(generator, **kw) -> Agent:
    return Agent(generator=generator, synthesizer=FakeSynthesizer("There are 7."), adapter=_Adapter(),
                 cache=TwoTierCache(L1Cache()), budget=Budget(), **kw)


def test_a_question_longer_than_500_characters_reaches_the_generator_whole():
    question = _question(1_200)
    prompt = user_prompt(_packet(question))
    assert "ENDOFQUESTION" in prompt, "the generator's question was cut"
    assert f"QUESTION: {question}" in prompt


def test_the_generator_and_the_judge_read_the_same_question():
    """The defect was a split: the judge read the whole question, the generator its first 500 characters."""
    question = _question(1_200)
    packet = _packet(question)
    seen = {}

    class _Judge:
        def score(self, q, schema, sql, preview):
            seen["question"] = q
            return 0.9

    sql = "SELECT count(*) AS n FROM claim"
    Verifier(sanity=False, judge=_Judge()).verify(packet, Approved(plan_sql=sql, target_sql=sql),
                                                  pa.table({"n": [7]}))
    generator_line = user_prompt(packet).splitlines()[0]
    assert generator_line == f"QUESTION: {seen['question']}"


def test_a_question_over_the_limit_is_declined_before_the_generator_is_called():
    generator = FakeGenerator([_REPLY])
    answer = _agent(generator, max_question_chars=1_000).answer(
        _packet(_question(1_001)), _snapshot(), _GRANTS, _IDENTITY)
    assert generator.calls == [], "the generator was asked a question over the limit"
    assert answer.deferred and not answer.failed
    assert answer.reason_code == DeferralReason.QUESTION_TOO_LONG
    assert "1,001" in answer.answer and "1,000" in answer.answer
    assert "MNEMIQ_MAX_QUESTION_CHARS" in answer.answer


def test_a_question_at_the_limit_is_answered():
    generator = FakeGenerator([_REPLY])
    answer = _agent(generator, max_question_chars=1_000).answer(
        _packet(_question(1_000)), _snapshot(), _GRANTS, _IDENTITY)
    assert len(generator.calls) == 1
    assert not answer.deferred, answer.answer


def test_an_agent_built_without_the_limit_still_enforces_the_shipped_default():
    """A construction site nobody has written yet, or one that forgets the argument, must not mean "no limit"."""
    generator = FakeGenerator([_REPLY])
    answer = _agent(generator).answer(_packet(_question(DEFAULT_MAX_QUESTION_CHARS + 1)), _snapshot(), _GRANTS,
                                      _IDENTITY)
    assert generator.calls == []
    assert answer.reason_code == DeferralReason.QUESTION_TOO_LONG


def test_the_setting_reads_its_env_var_and_defaults_to_the_shared_constant(monkeypatch):
    assert Settings.model_fields["max_question_chars"].default == DEFAULT_MAX_QUESTION_CHARS == 4_000
    monkeypatch.setenv("MNEMIQ_MAX_QUESTION_CHARS", "8000")
    assert Settings().max_question_chars == 8_000


def test_build_agent_carries_the_setting_to_the_agent():
    from mnemiq.agent.modes import MODES, build_agent

    for limit in (500, 8_000):
        agent = build_agent(MODES["thinking"], generator=None, synthesizer=None, adapter=None, cache=None,
                            corrector=None, values=None, selector=None, max_question_chars=limit)
        assert agent.max_question_chars == limit


def test_every_agent_construction_in_the_engine_passes_the_limit():
    """A source invariant, as for M35's flag: `eval.engine.build_engine` builds its own `Agent` and its only test
    caller needs a live key, so a behavioural walk cannot reach it. Every `Agent(...)` under `src/mnemiq` must pass
    `max_question_chars`, or that door enforces the default whatever the operator set."""
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "mnemiq"
    sites, missing = 0, []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Agent":
                sites += 1
                if "max_question_chars" not in {k.arg for k in node.keywords}:
                    missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert sites >= 2, f"found only {sites} Agent(...) construction sites -- this scan just went blind"
    assert missing == []


@pytest.mark.parametrize("limit", [500, 4_000])
def test_the_boot_window_check_budgets_the_configured_question(limit):
    import duckdb

    from mnemiq.llm import window

    from tests.test_prompt_window import _server, _settings, _snapshot as _window_snapshot

    report = window.check_window(duckdb.connect(), _window_snapshot(), _settings(max_question_chars=limit),
                                 "duckdb", http=_server({"count": 1_000, "max_model_len": 28_672}))
    assert report.prompt_tokens == 1_000 + limit
    assert f"{limit:,} tokens for the question" in report.sentence()
