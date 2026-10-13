"""M147: the question reaches the generator whole, or the agent declines it. It is never cut.

`user_prompt` wrote `QUESTION: {sanitize(packet.question, limit=500)}` while the judge, the candidate selector and the
synthesizer read the question whole, so a longer question was answered as a cut one and then checked against all of
it. On BEAVER's 200 sampled questions 84 reached the generator cut, and the annotations its harness appended to the
question almost never arrived (domain knowledge on 0 of 125 questions) while the judge read every word. Nothing said
so: the only witness was the model, which replied on 11 questions that the question looked truncated.

Now every stage reads the same question. Up to MNEMIQ_MAX_QUESTION_CHARS the generator gets it whole; past it both
doors decline before retrieval, which embeds the question, with a reason that says what to do (shorten it, or raise
the limit). `Agent.answer` declines it again, for a caller that skipped the door.
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
    """Exactly `chars` characters ending in a marker, so a cut anywhere is a missing tail."""
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


def test_the_generator_reads_backticks_and_line_breaks_as_the_judge_does():
    """M148: the generator alone read the question through `sanitize`, which turns backticks into quotes and folds line
    breaks, so sub-question SQL that quotes a column as `amount` reached it as the string 'amount' while the judge read
    the identifier."""
    question = "How much was paid?\n\nSub-question SQL: SELECT `amount` FROM claim"
    packet = _packet(question)
    seen = {}

    class _Judge:
        def score(self, q, schema, sql, preview):
            seen["question"] = q
            return 0.9

    sql = "SELECT count(*) AS n FROM claim"
    Verifier(sanity=False, judge=_Judge()).verify(packet, Approved(plan_sql=sql, target_sql=sql),
                                                  pa.table({"n": [7]}))
    assert seen["question"] == question
    assert user_prompt(packet).startswith(f"QUESTION: {question}\n")


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
            func = getattr(node, "func", None)
            named = getattr(func, "id", None) if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if isinstance(node, ast.Call) and named == "Agent":  # `Agent(...)` and `loop.Agent(...)` alike
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


def _fitter_case(question_chars: int, limit: int):
    from tests.test_prompt_fit import _fitter, _packet, _snapshot as _fit_snapshot

    counted = []

    def count(system, user):
        counted.append(len(user))
        return 10_000, 8_000  # always over: a question the fitter works on is counted, then trimmed

    snap = _fit_snapshot()
    fitter = _fitter(8_000, count=count)
    fitter.max_question_chars = limit
    packet = _packet(snap, question=_question(question_chars))
    return fitter, packet, snap, counted


def test_the_fitter_leaves_a_question_over_the_limit_alone(caplog):
    """Both doors fit before the agent: without this, a question the agent is about to decline is sent whole to the
    server's count, maybe more than once, and the operator's log says the prompt will not fit."""
    import logging

    from tests.test_prompt_fit import _grants

    fitter, packet, snap, counted = _fitter_case(1_001, 1_000)
    with caplog.at_level(logging.INFO, logger="mnemiq.semantic.fit"):
        out = fitter.fit(packet, snap, _grants(snap))
    assert out is packet and counted == []
    assert "prompt window" not in caplog.text


def test_the_fitter_still_fits_a_question_at_the_limit():
    from tests.test_prompt_fit import _grants

    fitter, packet, snap, counted = _fitter_case(1_000, 1_000)
    fitter.fit(packet, snap, _grants(snap))
    assert counted, "the control: a question within the limit is counted as before"


def test_fitter_for_carries_the_setting():
    from mnemiq.semantic.fit import fitter_for

    for limit in (500, 8_000):
        settings = Settings(max_question_chars=limit, llm_base_url="http://127.0.0.1:9/v1", llm_api_key="k",
                            llm_model="m")
        assert fitter_for(settings, "duckdb").max_question_chars == limit


# --- both doors decline before retrieval ---------------------------------------------------------
# Retrieval embeds the question (`embedder.embed([question])`), and the fitter counts it on the model
# server, so a check that waited for `Agent.answer` still sent an over-limit question to two servers
# and, if either failed, answered with that error instead of question_too_long (Codex's review of #102).

def _no_retrieve(*a, **k):
    raise AssertionError("retrieval ran: it embeds the question, so the embedding server was sent it")


class _NoFit:
    def fit(self, *a, **k):
        raise AssertionError("the fitter ran: it counts the question on the model server")


def test_runtime_declines_an_over_limit_question_before_retrieval_and_still_reports_it(monkeypatch):
    import mnemiq.runtime as rt_mod
    from mnemiq.runtime import Runtime

    monkeypatch.setattr(rt_mod, "retrieve", _no_retrieve)
    records, traces = [], []

    class _Sink:
        def record(self, source_id, record):
            records.append(record)

    class _Trace:
        def record_answer(self, event):
            traces.append(event)

    class _Authz:
        def grants_for(self, identity):
            return _GRANTS

    generator = FakeGenerator([_REPLY])
    rt = Runtime(con=None, snapshot=_snapshot(), adapter=None, agent=_agent(generator, max_question_chars=1_000),
                 embedder=None, authz=_Authz(), settings=None, sink=_Sink(), trace_sink=_Trace(), fitter=_NoFit())
    question = _question(1_001)
    answer = rt.ask(question, _IDENTITY)

    assert answer.reason_code == DeferralReason.QUESTION_TOO_LONG and answer.deferred and not answer.failed
    assert generator.calls == []
    assert answer.mode == "thinking" and answer.grant_fingerprint == _GRANTS.fingerprint
    assert [r.reason_code for r in records] == [DeferralReason.QUESTION_TOO_LONG]
    assert records[0].deferred and not records[0].failed
    assert [t.question for t in traces] == [question]


def test_the_eval_engine_declines_an_over_limit_question_before_retrieval(monkeypatch, tmp_path):
    import mnemiq.eval.engine as engine
    from types import SimpleNamespace

    monkeypatch.setattr(engine, "LLMEmbedder", lambda settings: None)
    monkeypatch.setattr(engine, "build_index", lambda con, *a, **k: con.execute(
        "CREATE TABLE IF NOT EXISTS semantic_object (object_id TEXT)"))
    monkeypatch.setattr(engine, "build_example_index", lambda *a, **k: None)
    monkeypatch.setattr(engine, "build_value_index", lambda *a, **k: 0)
    monkeypatch.setattr(engine, "retrieve", _no_retrieve)
    generator = FakeGenerator([_REPLY])
    monkeypatch.setattr(engine, "build_components", lambda settings, adapter, con: SimpleNamespace(
        fitter=_NoFit(), client=None, generator=generator, synthesizer=FakeSynthesizer("x"), corrector=None,
        values=None, selector=None))

    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
                        store_path=str(tmp_path / "s.duckdb"), max_question_chars=1_000)
    ask, _ = engine.build_engine(Snapshot(version="v", source_id="s", created_at="t"), adapter=None,
                                 settings=settings, grants=_GRANTS)
    answer = ask(_question(1_001))

    assert answer.reason_code == DeferralReason.QUESTION_TOO_LONG and answer.deferred and not answer.failed
    assert generator.calls == []


def test_runtime_declines_an_over_limit_question_before_the_snapshot_reload(monkeypatch):
    """The reload reaches the control database, and on a swap the model server's window count. Neither is sent the
    question, but either failing raised instead of the decline, and a swap's count held it up (Codex's review of
    #102's fix)."""
    import mnemiq.runtime as rt_mod
    from mnemiq.runtime import Runtime

    def _no_reload(*a, **k):
        raise AssertionError("the snapshot reload ran: it reaches the control database")

    monkeypatch.setattr(rt_mod, "load_current_snapshot", _no_reload)
    monkeypatch.setattr(rt_mod, "retrieve", _no_retrieve)
    records, traces = [], []

    class _Sink:
        def record(self, source_id, record):
            records.append(record)

    class _Trace:
        def record_answer(self, event):
            traces.append(event)

    class _Authz:
        def grants_for(self, identity):
            return _GRANTS

    settings = Settings(llm_base_url=None, llm_api_key=None, llm_model=None, pg_dsn="x", acme_data_dir=None,
                        control_dsn="postgresql://control.invalid/x")
    rt = Runtime(con=None, snapshot=_snapshot(), adapter=None, agent=_agent(FakeGenerator([_REPLY]),
                 max_question_chars=1_000), embedder=None, authz=_Authz(), settings=settings, sink=_Sink(),
                 trace_sink=_Trace(), fitter=_NoFit())
    answer = rt.ask(_question(1_001), _IDENTITY)

    assert answer.reason_code == DeferralReason.QUESTION_TOO_LONG
    assert [r.reason_code for r in records] == [DeferralReason.QUESTION_TOO_LONG]
    assert [t.packet for t in traces] == [None]
