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


# ---------------------------------------------------------------------------------------------
# M14 -- M6 split `failed` from `deferred`, and this serializer never learned. A failed answer
# went out as `deferred: false` with no `failed` key, so at the product surface an outage was
# indistinguishable from a successful answer -- strictly worse than the conflation M6 removed.
# ---------------------------------------------------------------------------------------------

from mnemiq.contract import DeferralReason


def test_a_failed_answer_is_distinguishable_from_an_answer():
    out = answer_payload(
        AgentAnswer(
            answer="Could not answer this question: the database rejected every attempt.",
            failed=True,
            reason_code=DeferralReason.EXECUTION_FAILED,
        )
    )

    assert out["failed"] is True, (
        "before this, a source outage serialized as deferred:false with no failed key -- "
        "a caller checking `deferred` saw false and had every reason to call it an answer"
    )
    assert out["deferred"] is False
    assert out["reason_code"] == "execution_failed"


def test_a_deferral_carries_the_reason_the_deferral_card_needs():
    out = answer_payload(
        AgentAnswer(
            answer="Answering this would require access to 'salary'.",
            deferred=True,
            reason_code=DeferralReason.AUTHORIZATION,
        )
    )

    assert out["deferred"] is True and out["failed"] is False
    assert out["reason_code"] == "authorization", (
        "the UI section asks the DeferralCard to render a D5 reason state; the wire has to carry it"
    )


def test_a_plain_answer_says_so_in_both_fields():
    out = answer_payload(AgentAnswer(answer="2.", trace=_trace()))

    assert out["failed"] is False
    assert out["reason_code"] is None


def test_the_field_is_spelled_the_way_mcp_spells_it():
    # One spelling across MCP, /v1/ask and SSE. `deferral_reason` would also be wrong on its own
    # terms: EXECUTION_FAILED is not a deferral, and naming the field after one would re-merge the
    # two states M6 separated.
    out = answer_payload(AgentAnswer(answer="x", deferred=True,
                                     reason_code=DeferralReason.NO_TABLES))

    assert "reason_code" in out and "deferral_reason" not in out
