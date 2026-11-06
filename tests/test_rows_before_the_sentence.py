"""The rows go out as soon as they are verified, and the sentence follows.

The prose call turns the rows into a sentence, and on a local model it is a large share of an answer's time. A reader
can read the table while it runs, so the engine sends the verified result first, on the stream, carrying everything the
final answer will say about those rows: the SQL as asked, the preview and the governed notes. Built once, so the early
rows and the final answer cannot differ. A prose call that fails after the rows went out leaves the rows standing,
and says so.
"""

from __future__ import annotations

import dataclasses
import json

import pyarrow as pa
from fastapi.testclient import TestClient

from mnemiq.agent.loop import AgentAnswer, cap_sentence
from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.contract import Column, IdentityContext, Snapshot
from mnemiq.contract.seams import append_notes
from mnemiq.generate.generator import FakeGenerator
from mnemiq.llm.client import ModelUnavailable, PromptCut
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.server.app import build_app
from mnemiq.server.serialize import answer_payload, result_payload

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_GOOD = '{"sql": "SELECT n FROM claim", "reason": "r"}'


class _Adapter:
    dialect = "duckdb"

    def __init__(self, table=None):
        self.table = table if table is not None else pa.table({"n": [7]})

    def execute_arrow(self, sql, timeout_s=None):
        return self.table

    def execute(self, sql):
        return []


class _FailingSynthesizer:
    def __init__(self, exc):
        self.exc = exc

    def answer(self, *args, **kwargs):
        raise self.exc


def _snapshot() -> Snapshot:
    return Snapshot(version="v1", source_id="acme", created_at="t", columns=[Column(id="claim.n", object_id="claim", name="n")])


def _packet() -> ContextPacket:
    return ContextPacket(question="how many claims?", cards=[RetrievedCard(object_id="claim", card="TABLE claim", score=1.0)],
                         grant_fingerprint=_GRANTS.fingerprint, enrichment_version="v1")


def _ask(replies=(_GOOD,), synth=None, table=None, mode="instant"):
    events: list[dict] = []
    agent = build_agent(MODES[mode], generator=FakeGenerator(list(replies)), synthesizer=synth or FakeSynthesizer("prose"),
                        adapter=_Adapter(table), cache=TwoTierCache(L1Cache()), corrector=None, values=None,
                        selector=None)
    answer = agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY, emit=events.append)
    return answer, events


def _results(events):
    return [e for e in events if e.get("kind") == "result"]


def _first(events, predicate):
    return next(i for i, e in enumerate(events) if predicate(e))


# --- the engine --------------------------------------------------------------------------------------------------------


def test_the_rows_go_out_before_the_prose_starts():
    _, events = _ask()
    assert len(_results(events)) == 1
    result_at = _first(events, lambda e: e.get("kind") == "result")
    prose_at = _first(events, lambda e: e.get("stage") == "synthesize" and e.get("phase") == "started")
    verified_at = _first(events, lambda e: e.get("stage") == "verify" and e.get("phase") == "finished")
    assert verified_at < result_at < prose_at


def test_the_early_rows_are_the_final_answers():
    answer, events = _ask(table=pa.table({"n": list(range(1000))}))   # fills the guard's limit, so a note rides along
    early, final = result_payload(_results(events)[0]), answer_payload(answer)
    for field in ("sql", "tables_used", "narrowed", "preview", "lineage"):
        assert early[field] == final[field], field
    assert early["notes"], "the case: a governed note to carry"
    assert cap_sentence(1000) in early["notes"]
    assert final["answer"] == append_notes("prose", *early["notes"])


def test_the_early_rows_carry_no_prose():
    _, events = _ask()
    assert "answer" not in result_payload(_results(events)[0])


def test_a_declined_answer_sends_no_rows():
    answer, events = _ask(replies=['{"sql": null, "reason": "no such table"}'])
    assert answer.deferred and _results(events) == []


def test_without_prose_the_rows_still_go_out_first():
    events: list[dict] = []
    mode = dataclasses.replace(MODES["instant"], prose=False)
    agent = build_agent(mode, generator=FakeGenerator([_GOOD]), synthesizer=FakeSynthesizer("prose"), adapter=_Adapter(),
                        cache=TwoTierCache(L1Cache()), corrector=None, values=None, selector=None)
    agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY, emit=events.append)
    assert len(_results(events)) == 1


def test_the_rows_time_is_stamped_before_the_total():
    answer, _ = _ask()
    timing = answer.trace.timing
    assert 0 <= timing["rows_ms"] <= timing["total_ms"]


def test_with_no_listener_the_answer_is_unchanged():
    agent = build_agent(MODES["instant"], generator=FakeGenerator([_GOOD]), synthesizer=FakeSynthesizer("prose"),
                        adapter=_Adapter(), cache=TwoTierCache(L1Cache()), corrector=None, values=None, selector=None)
    answer = agent.answer(_packet(), _snapshot(), _GRANTS, _IDENTITY)
    assert answer.answer == "prose" and not answer.failed and not answer.prose_failed


# --- a prose call that fails after the rows ----------------------------------------------------------------------------


def test_a_prose_failure_leaves_the_rows_standing_and_says_why():
    answer, events = _ask(synth=_FailingSynthesizer(ModelUnavailable("upstream said 503")))
    assert not answer.failed and not answer.deferred
    assert answer.prose_failed
    assert answer.answer.startswith("Returned 1 row.")
    assert "upstream said 503" in answer.answer
    assert answer.preview is not None and answer.preview.rows
    assert len(_results(events)) == 1
    assert answer_payload(answer)["prose_failed"] is True


def test_a_cut_prompt_to_the_prose_model_is_stated_too():
    answer, _ = _ask(synth=_FailingSynthesizer(PromptCut("the server kept 2,048 tokens of 9,000")))
    assert answer.prose_failed and "2,048 tokens" in answer.answer


def test_the_notes_still_follow_a_failed_sentence():
    answer, _ = _ask(synth=_FailingSynthesizer(ModelUnavailable("down")), table=pa.table({"n": list(range(1000))}))
    assert answer.answer.endswith(cap_sentence(1000))


def test_an_answer_with_prose_is_not_marked():
    answer, _ = _ask()
    assert answer.prose_failed is False and answer_payload(answer)["prose_failed"] is False


# --- the stream --------------------------------------------------------------------------------------------------------


class _RT:
    """Replays a real agent's events and answer through the chat door."""

    def __init__(self, table=None):
        self.table = table

    def ask(self, question, identity, mode=None, emit=None, history=None, profile=None):
        answer, events = _ask(table=self.table)
        for event in events if emit else ():
            emit(event)
        return answer


def _frames(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def test_the_stream_sends_the_rows_before_any_text():
    r = TestClient(build_app(_RT(), IdentityContext(tenant_id="t", principal_id="u", roles=[]))).post(
        "/v1/chat", json={"question": "q"})
    frames = _frames(r.text)
    names = [(f["type"], f.get("name"), f.get("stepName")) for f in frames]
    rows_at = names.index(("CUSTOM", "mnemiq.result", None))
    assert rows_at < names.index(("TEXT_MESSAGE_START", None, None))
    assert rows_at < next(i for i, n in enumerate(names) if n == ("STEP_STARTED", None, "synthesize"))
    early = frames[rows_at]["value"]
    final = next(f for f in frames if f.get("name") == "mnemiq.answer")["value"]
    assert early["preview"] == final["preview"] and early["sql"] == final["sql"]


def test_an_answer_without_rows_streams_as_before():
    rt = type("RT", (), {"ask": lambda self, *a, **k: AgentAnswer(answer="2 claims.", mode="thinking")})()
    r = TestClient(build_app(rt, IdentityContext(tenant_id="t", principal_id="u", roles=[]))).post(
        "/v1/chat", json={"question": "q"})
    assert [f["type"] for f in _frames(r.text)] == ["RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT",
                                                     "TEXT_MESSAGE_END", "CUSTOM", "RUN_FINISHED"]


def test_ask_reports_when_the_rows_were_ready():
    r = TestClient(build_app(_RT(), IdentityContext(tenant_id="t", principal_id="u", roles=[]))).post(
        "/v1/ask", json={"question": "q"})
    timing = r.json()["timing"]
    assert 0 <= timing["rows_ms"] <= timing["total_ms"]


def test_the_answer_log_records_a_failed_sentence(tmp_path):
    from mnemiq.observability.answer_log import AnswerLog

    answer, _ = _ask(synth=_FailingSynthesizer(ModelUnavailable("down")))
    log = AnswerLog(str(tmp_path / "answers.jsonl"))
    log.record_answer(answer, "how many claims?", 1.0)
    record = json.loads((tmp_path / "answers.jsonl").read_text().splitlines()[-1])
    assert record["prose_failed"] is True and record["failed"] is False


def test_under_a_row_filter_the_early_rows_show_the_sql_as_asked():
    """M51 holds for the early rows too: the policy's rewrite names tables and literals this caller is never granted."""
    from mnemiq.agent.trace import build_trace
    from mnemiq.sql.decide import decide
    from mnemiq.sql.policy import build_access_policy

    snapshot = Snapshot(version="v1", source_id="s", created_at="t", columns=[
        Column(id=f"{o}.{n}", object_id=o, name=n)
        for o, n in [("claim", "id"), ("entitlement", "id"), ("entitlement", "who")]])
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": "id IN (SELECT id FROM entitlement WHERE who = 'u')"})
    approved = decide("SELECT id FROM claim", {"claim": {"id"}}, policy=build_access_policy(snapshot, grants),
                      target="duckdb")
    assert "entitlement" in approved.target_sql  # the control: the policy really applied
    trace = build_trace("q", approved, _IDENTITY, "v1", {"rows_ms": 1.0}, "1x1", [])
    early = result_payload({"trace": trace, "preview": None, "notes": []})
    assert early["sql"] == "SELECT id FROM claim LIMIT 1000" and early["narrowed"]
