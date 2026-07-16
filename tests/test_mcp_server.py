from mnemiq.agent.loop import AgentAnswer
from mnemiq.contract import IdentityContext, Trace
from mnemiq.mcp.server import _db_read, _get_schema


def _identity():
    return IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])


def _trace():
    return Trace(question="q", plan_sql="SELECT 1", target_sql="SELECT count(*) FROM claim",
                 result_shape="scalar", timing={"total_ms": 5.0}, enrichment_version="v1",
                 identity=_identity(), tables_used=["claim"])


class _RT:
    def __init__(self, answer, cards):
        self._answer, self._cards = answer, cards
        self.mode = "UNSET"

    def ask(self, question, identity, mode=None):
        self.mode = mode
        return self._answer

    def schema(self, identity):
        return self._cards


def test_db_read_returns_answer_sql_trace_and_not_deferred():
    rt = _RT(AgentAnswer(answer="2 claims.", trace=_trace(), deferred=False), [])
    out = _db_read(rt, _identity(), "how many claims?")
    assert out["answer"] == "2 claims."
    assert out["sql"] == "SELECT count(*) FROM claim"
    assert out["deferred"] is False
    assert out["trace"]["tables_used"] == ["claim"]


def test_db_read_passes_a_deferral_through_honestly():
    rt = _RT(AgentAnswer(answer="No table holds salary.", deferred=True), [])
    out = _db_read(rt, _identity(), "avg salary?")
    assert out["deferred"] is True
    assert out["sql"] is None  # nothing ran; never a fabricated query


def test_get_schema_returns_the_granted_cards():
    rt = _RT(None, [{"object_id": "claim", "card": "TABLE claim ..."}])
    out = _get_schema(rt, _identity())
    assert out["tables"] == [{"object_id": "claim", "card": "TABLE claim ..."}]


def test_db_read_passes_mode_through_and_echoes_it():
    ans = AgentAnswer(answer="2 claims.", trace=_trace(), deferred=False, mode="deep")
    rt = _RT(ans, [])
    out = _db_read(rt, _identity(), "how many claims?", mode="deep")
    assert rt.mode == "deep"
    assert out["mode"] == "deep"


def test_db_read_defaults_mode_to_none_so_the_router_decides():
    rt = _RT(AgentAnswer(answer="2 claims.", trace=_trace(), deferred=False, mode="thinking"), [])
    out = _db_read(rt, _identity(), "how many claims?")
    assert rt.mode is None
    assert out["mode"] == "thinking"  # what the router resolved, echoed back
