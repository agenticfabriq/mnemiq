from mnemiq.contract import IdentityContext, Trace


def test_identity_defaults():
    i = IdentityContext(tenant_id="t1", principal_id="u1")
    assert i.groups == [] and i.attributes == {}


def test_trace_round_trip_with_identity():
    t = Trace(
        question="total revenue?",
        plan_sql="SELECT ...",
        target_sql="SELECT ...",
        result_shape="scalar",
        timing={"total_ms": 12.0},
        enrichment_version="v1",
        identity=IdentityContext(tenant_id="t1", principal_id="u1", groups=["analyst"]),
        tables_used=["v_revenue"],
    )
    assert Trace.model_validate_json(t.model_dump_json()) == t
