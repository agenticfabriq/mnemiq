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


def test_db_read_carries_the_bounded_preview():
    from mnemiq.agent.loop import ResultPreview

    p = ResultPreview(columns=["n"], rows=[[2]], row_count=1, truncated=False)
    rt = _RT(AgentAnswer(answer="2 claims.", trace=_trace(), preview=p), [])
    out = _db_read(rt, _identity(), "how many claims?")
    assert out["preview"] == {"columns": ["n"], "rows": [[2]], "row_count": 1,
                              "truncated": False}


def test_db_read_preview_is_none_on_deferral():
    rt = _RT(AgentAnswer(answer="No table holds salary.", deferred=True), [])
    assert _db_read(rt, _identity(), "avg salary?")["preview"] is None


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


def test_db_write_refused_by_default():
    from mnemiq.mcp.server import _db_write
    from mnemiq.runtime import WriteResult

    class _RT:
        def write(self, sql, identity):
            return WriteResult(approved=False, refusal="You may not write to 'claim'.",
                               target="claim")

    out = _db_write(_RT(), _identity(), "INSERT INTO claim (id) VALUES (1)")
    assert out["approved"] is False and "may not write" in out["refusal"]
    assert out["rows_affected"] is None


def test_db_write_reports_rows_on_approval():
    from mnemiq.mcp.server import _db_write
    from mnemiq.runtime import WriteResult

    class _RT:
        def write(self, sql, identity):
            return WriteResult(approved=True, target="claim", rows_affected=1,
                               target_sql="INSERT INTO claim ...")

    out = _db_write(_RT(), _identity(), "INSERT INTO claim (id) VALUES (1)")
    assert out["approved"] is True and out["target"] == "claim" and out["rows_affected"] == 1
