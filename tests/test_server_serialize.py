from mnemiq.agent.loop import AgentAnswer, ResultPreview
from mnemiq.contract import IdentityContext, Trace
from mnemiq.server.serialize import answer_payload


def _trace():
    ident = IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"])
    return Trace(question="q", plan_sql="SELECT 1", target_sql="SELECT count(*) FROM claim",
                 result_shape="scalar", timing={"total_ms": 5.0}, enrichment_version="v1",
                 identity=ident, tables_used=["claim"])


def test_full_answer_payload():
    p = ResultPreview(columns=["n"], rows=[[2]], row_count=1, truncated=False)
    out = answer_payload(AgentAnswer(answer="2.", trace=_trace(), cached=True,
                                     agreement=0.8, mode="deep", preview=p))
    assert out["answer"] == "2."
    assert out["sql"] == "SELECT count(*) FROM claim"
    assert out["tables_used"] == ["claim"]
    assert out["timing"] == {"total_ms": 5.0}
    assert out["cached"] is True and out["agreement"] == 0.8 and out["mode"] == "deep"
    assert out["preview"]["rows"] == [[2]]


def test_deferral_payload_has_no_sql_no_preview():
    out = answer_payload(AgentAnswer(answer="No table holds salary.", deferred=True))
    assert out["deferred"] is True
    assert out["sql"] is None and out["preview"] is None and out["tables_used"] is None
