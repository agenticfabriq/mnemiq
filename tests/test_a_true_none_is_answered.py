"""M146: a question whose true answer is "none" is answered, through the doors a caller uses.

The no-rows rule declined every empty result, so a correct "none" was unanswerable in every mode,
and the judge never saw one. An empty result is answered now when every string value it filters
on is grounded in the value index; one resting on a value nothing could check is still declined,
and the decline names that value. Driven here through the real Agent with the product's own
per-mode verifiers (`_build_mode_verifiers`, `build_agent`) over a synthetic DuckDB source that the
real structural enrichment profiles and the real value index covers, and through `Runtime.ask` to
the payload `/v1/ask` serves. Only the model calls are scripted.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from mnemiq.adapters.duckdb import DuckDBAdapter
from mnemiq.agent.modes import MODES, build_agent
from mnemiq.agent.synthesize import FakeSynthesizer
from mnemiq.authz.grants import GrantSet
from mnemiq.cache.store import L1Cache, TwoTierCache
from mnemiq.config import Settings
from mnemiq.contract import IdentityContext
from mnemiq.enrichment.pipeline import enrich_structural
from mnemiq.generate.generator import FakeGenerator
from mnemiq.runtime import _build_mode_verifiers
from mnemiq.semantic.retrieval import ContextPacket, RetrievedCard
from mnemiq.semantic.values import ValueIndex, build_value_index
from mnemiq.server.serialize import answer_payload
from mnemiq.store.bootstrap import init_store

_GRANTS = GrantSet(frozenset({"claim"}))
_IDENTITY = IdentityContext(tenant_id="t1", principal_id="u1", roles=["analyst"])
_CARD = ("TABLE claim: id INTEGER, region VARCHAR (north|south|east|west), status VARCHAR "
         "(open|closed), amount DOUBLE, adjuster VARCHAR, filed DATE")

# The true answer is "none": no claim is anywhere near a million (asserted below, not assumed).
NONE_SQL = "SELECT id, amount FROM claim WHERE status = 'open' AND amount > 1000000"
NONE_NOTE = "No rows matched: status = 'open' AND amount > 1000000."
# A misspelled value on a column with 300 distinct values, past the index's 200: nothing checks it
# before the run, so it returns nothing, and "none" would be a wrong answer.
MISSPELLED_SQL = "SELECT id, amount FROM claim WHERE adjuster = 'agnet_0042'"
MISSPELLED_DECLINE = ("No rows matched, and 'agnet_0042' could not be checked against "
                      "claim.adjuster's known values")


class _CountingJudgeClient:
    """The endpoint behind the real `SemanticJudge`: approves, and records what it was shown."""

    def __init__(self, reply: str = '{"confidence": 0.95}'):
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, system, user, max_tokens=None, temperature=None, **_):
        self.calls.append(user)
        return self.reply


@pytest.fixture
def source(tmp_path, monkeypatch):
    for var in ("MNEMIQ_VERIFY", "MNEMIQ_VERIFY_SANITY", "MNEMIQ_VERIFY_BASE_URL",
                "MNEMIQ_VERIFY_MODEL", "MNEMIQ_VERIFY_API_KEY", "MNEMIQ_VERIFY_GROUNDING",
                "MNEMIQ_VERIFY_THRESHOLD", "MNEMIQ_VERIFY_FAIL_CLOSED"):
        monkeypatch.delenv(var, raising=False)
    path = str(tmp_path / "syn.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE claim (id INTEGER, region VARCHAR, status VARCHAR, amount DOUBLE, "
                "adjuster VARCHAR, filed DATE)")
    regions = ["north", "south", "east", "west"]
    rows = [(i, regions[i % 4], "open" if i % 3 else "closed", float(100 + (i * 37) % 4900),
             f"agent_{i % 300:04d}", f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}")
            for i in range(1200)]
    con.executemany("INSERT INTO claim VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.close()
    adapter = DuckDBAdapter.duckdb(path)
    snap = enrich_structural(adapter, "syn")
    store = init_store(str(tmp_path / "store.duckdb"))
    build_value_index(adapter, snap, store)
    values = ValueIndex(store)
    # The premises, measured rather than assumed.
    assert values.has("claim", "status") and values.has("claim", "region")
    assert not values.has("claim", "adjuster")
    assert adapter.execute("SELECT max(amount) FROM claim")[0][0] < 1_000_000
    assert adapter.execute("SELECT count(*) FROM claim WHERE adjuster = 'agent_0042'")[0][0] > 0
    return adapter, snap, values


def _agents(adapter, values, sql: str, judge_client, settings: Settings | None = None):
    verifiers, _ = _build_mode_verifiers(settings or Settings(), client=judge_client)
    reply = json.dumps({"sql": sql})
    return {name: build_agent(mode, generator=FakeGenerator([reply] * 6),
                              synthesizer=FakeSynthesizer("SYNTHESIZED"), adapter=adapter,
                              cache=TwoTierCache(L1Cache()), corrector=None, values=values,
                              selector=None, verifier=verifiers[name])
            for name, mode in MODES.items()}


def _packet(question: str, grants: GrantSet = _GRANTS) -> ContextPacket:
    return ContextPacket(question=question,
                         cards=[RetrievedCard(object_id="claim", card=_CARD, score=1.0)],
                         grant_fingerprint=grants.fingerprint, enrichment_version="v1")


def test_a_true_none_is_answered_in_every_mode_and_the_judge_reads_it_where_one_is_wired(source):
    adapter, snap, values = source
    judge = _CountingJudgeClient()
    agents = _agents(adapter, values, NONE_SQL, judge)
    for name in ("instant", "thinking", "deep"):
        before = len(judge.calls)
        ans = agents[name].answer(_packet("Which open claims are over 1,000,000?"), snap,
                                  _GRANTS, _IDENTITY)
        assert ans.deferred is False and ans.failed is False, (name, ans.answer)
        assert ans.preview is not None and ans.preview.row_count == 0, name
        # The note is the engine's, appended after the synthesizer and never through it.
        assert ans.answer.startswith("SYNTHESIZED"), name
        assert NONE_NOTE in ans.answer, (name, ans.answer)
        judged = len(judge.calls) - before
        if name == "instant":
            assert judged == 0 and ans.verify_layer == "pass"  # the grounding is the whole decision
        else:
            # A grounded empty result goes on to the judge, which reads the zero rows.
            assert judged == 1 and ans.verify_layer == "judge", name
            assert "(0 rows)" in judge.calls[-1]


def test_a_value_nothing_could_check_keeps_the_empty_result_declined_and_is_named(source):
    adapter, snap, values = source
    judge = _CountingJudgeClient()
    agents = _agents(adapter, values, MISSPELLED_SQL, judge)
    for name in ("instant", "thinking", "deep"):
        ans = agents[name].answer(_packet("List the claims handled by adjuster agent_0042."), snap,
                                  _GRANTS, _IDENTITY)
        assert ans.deferred is True and ans.failed is False, (name, ans.answer)
        assert ans.verify_layer == "sanity" and str(ans.reason_code) == "verification", name
        assert ans.answer.startswith(MISSPELLED_DECLINE), (name, ans.answer)
    assert judge.calls == []  # declined before the judge: no model call spent on it


def test_a_none_asked_as_a_total_is_answered_as_one(source):
    """`SUM` over no rows is one NULL, which the null-scalar sibling rule declined (M146)."""
    adapter, snap, values = source
    sql = "SELECT sum(amount) AS total FROM claim WHERE status = 'open' AND amount > 1000000"
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    ans = agents["instant"].answer(_packet("What is the total of open claims over 1,000,000?"),
                                   snap, _GRANTS, _IDENTITY)
    assert ans.deferred is False, ans.answer
    assert ("The result holds no value: no rows matched (status = 'open' AND amount > 1000000), "
            "or every row that matched holds none.") in ans.answer


def test_a_misspelled_total_stays_declined(source):
    adapter, snap, values = source
    sql = "SELECT sum(amount) AS total FROM claim WHERE adjuster = 'agnet_0042'"
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    ans = agents["instant"].answer(_packet("What is the total for adjuster agent_0042?"),
                                   snap, _GRANTS, _IDENTITY)
    assert ans.deferred is True and ans.verify_layer == "sanity"
    assert ans.answer.startswith("The query produced an empty value, and 'agnet_0042' could not "
                                 "be checked against claim.adjuster's known values"), ans.answer


def test_a_date_range_needs_no_grounding(source):
    """A well-formed date against a DATE column is a bound, as a number is: no spelling to check."""
    adapter, snap, values = source
    sql = "SELECT id FROM claim WHERE region = 'west' AND filed >= '2031-01-01'"
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    ans = agents["instant"].answer(_packet("Which west claims were filed in 2031 or later?"),
                                   snap, _GRANTS, _IDENTITY)
    assert ans.deferred is False, ans.answer
    assert "No rows matched: region = 'west' AND filed >= '2031-01-01'." in ans.answer


def test_a_row_filtered_callers_true_none_is_answered_with_the_narrowing_note(source):
    """Under a row filter an empty result is often the true answer for that caller (M117). The
    value index is built without the filter (M5), so a grounded value is one the table holds."""
    adapter, snap, values = source
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": "region = 'north'"})
    agents = _agents(adapter, values, "SELECT id, amount FROM claim WHERE region = 'south'",
                     _CountingJudgeClient())
    ans = agents["instant"].answer(_packet("List the south-region claims.", grants), snap, grants,
                                   _IDENTITY)
    assert ans.deferred is False, ans.answer
    assert ans.narrowed and ans.narrowed[0].rows is True
    assert "No rows matched: region = 'south'." in ans.answer
    assert "Row filter applied to: claim." in ans.answer
    # The note names the statement as asked, never the policy's own predicate.
    assert "north" not in ans.answer


# --- through Runtime.ask, to the payload /v1/ask serves -------------------------------------------

def _runtime(monkeypatch, adapter, snap, values, sql: str, grants: GrantSet = _GRANTS):
    import mnemiq.runtime as rt_mod
    from mnemiq.runtime import Runtime

    class _Authz:
        def grants_for(self, _identity):
            return grants

    def _retrieve(con, question, *a, **k):
        return _packet(question, grants)

    monkeypatch.setattr(rt_mod, "retrieve", _retrieve)
    agents = _agents(adapter, values, sql, _CountingJudgeClient())
    return Runtime(con=None, snapshot=snap, adapter=adapter, agent=agents["thinking"],
                   embedder=None, authz=_Authz(), settings=None, agents=agents)


def test_runtime_ask_serves_a_true_none_as_an_answer(source, monkeypatch):
    adapter, snap, values = source
    rt = _runtime(monkeypatch, adapter, snap, values, NONE_SQL)
    for mode in ("instant", "thinking"):
        payload = answer_payload(rt.ask("Which open claims are over 1,000,000?", _IDENTITY,
                                        mode=mode))
        assert payload["deferred"] is False and payload["failed"] is False, payload["answer"]
        assert payload["preview"]["row_count"] == 0 and payload["preview"]["rows"] == []
        assert NONE_NOTE in payload["answer"]
        assert payload["verified"] == ("basic" if mode == "instant" else "judged")


def test_runtime_ask_declines_a_misspelled_none_and_names_the_value(source, monkeypatch):
    adapter, snap, values = source
    rt = _runtime(monkeypatch, adapter, snap, values, MISSPELLED_SQL)
    payload = answer_payload(rt.ask("List the claims handled by adjuster agent_0042.", _IDENTITY,
                                    mode="instant"))
    assert payload["deferred"] is True and payload["reason_code"] == "verification"
    assert payload["verified"] == "basic" and payload["preview"] is None
    assert payload["answer"].startswith(MISSPELLED_DECLINE)


def test_runtime_ask_answers_a_row_filtered_none_with_the_narrowing(source, monkeypatch):
    adapter, snap, values = source
    grants = GrantSet(frozenset({"claim"}), row_filters={"claim": "region = 'north'"})
    rt = _runtime(monkeypatch, adapter, snap, values,
                  "SELECT id, amount FROM claim WHERE region = 'south'", grants)
    payload = answer_payload(rt.ask("List the south-region claims.", _IDENTITY, mode="instant"))
    assert payload["deferred"] is False, payload["answer"]
    assert payload["narrowed"] == [{"object": "claim", "rows": True, "columns": False}]
    assert "No rows matched: region = 'south'." in payload["answer"]
    assert "Row filter applied to: claim." in payload["answer"]
